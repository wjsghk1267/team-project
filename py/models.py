import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
from config import BREED_MAPPING
from collections import Counter

def process_yolo_results(results):
    lstm_keypoint_sequence = []
    skeleton_sequence = []
    breed_counter = Counter()

    for r in results:
        if r.keypoints is not None and len(r.keypoints) > 0:
            yolo_keypoints = r.keypoints[0].cpu().numpy()
            lstm_keypoints = convert_yolo_to_lstm(yolo_keypoints)
            lstm_keypoint_sequence.append(lstm_keypoints)
            skeleton = create_skeleton(lstm_keypoints)
            skeleton_sequence.append(skeleton)

            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = r.names[cls]
                breed_counter[class_name] += 1

    return np.array(lstm_keypoint_sequence), np.array(skeleton_sequence), breed_counter

def convert_yolo_to_lstm(yolo_keypoints):
    image_width, image_height = 640, 384
    yolo_to_lstm_mapping = {
        16: 0, 23: 4, 8: 5, 2: 6, 7: 7, 1: 8, 10: 9, 4: 10, 9: 11, 3: 12, 12: 13, 13: 14
    }
    lstm_keypoints = np.zeros((15, 2), dtype=float)
    
    for yolo_index, lstm_index in yolo_to_lstm_mapping.items():
        if yolo_index < yolo_keypoints.shape[0]:
            x, y = yolo_keypoints[yolo_index]
            lstm_keypoints[lstm_index] = [x / image_width, y / image_height]
    
    if 20 < yolo_keypoints.shape[0]:
        lstm_keypoints[1] = yolo_keypoints[20] / np.array([image_width, image_height])
    if 17 < yolo_keypoints.shape[0]:
        lstm_keypoints[2] = lstm_keypoints[3] = yolo_keypoints[17] / np.array([image_width, image_height])
    
    return lstm_keypoints

def create_skeleton(keypoints):
    DOG_SKELETON = [
        [0, 1], [0, 2], [2, 3], [1, 4], [4, 5], [4, 6], [5, 7], [6, 8],
        [9, 11], [10, 12], [4, 13], [13, 14], [9, 13], [10, 13], [5, 9],
        [6, 10], [5, 6], [9, 10]
    ]
    skeleton = []
    for start, end in DOG_SKELETON:
        start_point, end_point = keypoints[start], keypoints[end]
        skeleton.extend([start_point[0], start_point[1], end_point[0], end_point[1]])
    return skeleton

class ImprovedLSTMModel(nn.Module):
    def __init__(self, keypoint_size, skeleton_size, hidden_size, num_layers, num_classes):
        super(ImprovedLSTMModel, self).__init__()
        self.keypoint_lstm = nn.LSTM(keypoint_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.skeleton_lstm = nn.LSTM(skeleton_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, keypoints, skeleton):
        _, (h_n_keypoints, _) = self.keypoint_lstm(keypoints)
        _, (h_n_skeleton, _) = self.skeleton_lstm(skeleton)
        combined = torch.cat((h_n_keypoints[-1], h_n_skeleton[-1]), dim=1)
        out = self.dropout(combined)
        out = self.fc(out)
        return out

def load_yolo_model(model_path):
    return YOLO(model_path)

def load_lstm_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    keypoint_size = checkpoint.get('keypoint_size', 30)
    skeleton_size = checkpoint.get('skeleton_size', 30)
    hidden_size = checkpoint.get('hidden_size', 64)
    num_layers = checkpoint.get('num_layers', 2)
    num_classes = checkpoint.get('num_classes', 10)
    
    model = ImprovedLSTMModel(keypoint_size, skeleton_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint.get('all_class_names', []), checkpoint.get('metadata', [])

def get_most_common_breed(breed_counter):
    if breed_counter:
        most_common_breed = breed_counter.most_common(1)[0][0]
        total_detections = sum(breed_counter.values())
        breed_percentage = (breed_counter[most_common_breed] / total_detections) * 100
        return most_common_breed, breed_percentage
    return "믹스", 0

def standardize_sequence_length(lstm_keypoint_sequence, skeleton_sequence, target_length=100):
    if len(lstm_keypoint_sequence) > target_length:
        lstm_keypoint_sequence = lstm_keypoint_sequence[:target_length]
        skeleton_sequence = skeleton_sequence[:target_length]
    elif len(lstm_keypoint_sequence) < target_length:
        padding = [np.zeros_like(lstm_keypoint_sequence[-1])] * (target_length - len(lstm_keypoint_sequence))
        lstm_keypoint_sequence = np.concatenate([lstm_keypoint_sequence, padding], axis=0)
        skeleton_padding = [np.zeros_like(skeleton_sequence[-1])] * (target_length - len(skeleton_sequence))
        skeleton_sequence = np.concatenate([skeleton_sequence, skeleton_padding], axis=0)
    return lstm_keypoint_sequence, skeleton_sequence

def predict_breed(results, conf_thresh):
    breed_counter = Counter()
    for r in results:
        for c in r.boxes.cls:
            class_name = r.names[int(c)]
            if class_name in BREED_MAPPING and r.boxes.conf[0] > conf_thresh:
                breed_counter[BREED_MAPPING[class_name]] += 1
    
    if not breed_counter:
        return "알 수 없음", 0

    total = sum(breed_counter.values())
    most_common = breed_counter.most_common(1)[0]
    return most_common[0], (most_common[1] / total) * 100