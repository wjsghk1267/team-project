from collections import Counter
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import tqdm
from config import font_path

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

            # YOLO 결과에서 품종 정보 추출 및 카운트
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = r.names[cls]
                breed_counter[class_name] += 1  # 품종 등장 횟수 증가

    return np.array(lstm_keypoint_sequence), np.array(skeleton_sequence), breed_counter

# Define function to get most common breed
def get_most_common_breed(breed_counter):
    if breed_counter:
        most_common_breed = breed_counter.most_common(1)[0][0]
        total_detections = sum(breed_counter.values())
        breed_percentage = (breed_counter[most_common_breed] / total_detections) * 100
        return most_common_breed, breed_percentage
    return "믹스", 0

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
    
    # 이마, 입꼬리, 아래 입술 중앙 처리 (가능한 경우에만)
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

# YOLO 결과 처리 및 LSTM 입력 준비
def process_video(video_path, lstm_text, yolo_text):
    if not os.path.exists(video_path):
        print(f"오류: 비디오 파일을 찾을 수 없습니다: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 비디오 파일을 열 수 없습니다: {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = 'output_video_with_labels.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 한글 폰트 설정
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)

    try:
        for _ in tqdm(range(total_frames), desc="비디오 처리 중"):
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽는 데 실패했습니다.")
                break

            # OpenCV 이미지를 Pillow 이미지로 변환
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            # YOLO 텍스트 추가
            y = 30
            for line in yolo_text.split(', '):
                draw.text((10, y), line, font=font, fill=(255, 255, 255))
                y += 30

            # LSTM 텍스트 추가
            y += 30
            for line in lstm_text.split(', '):
                draw.text((10, y), line, font=font, fill=(255, 255, 255))
                y += 30

            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            out.write(frame)

    except Exception as e:
        print(f"비디오 처리 중 오류 발생: {str(e)}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    if os.path.exists(output_path):
        print(f"레이블이 추가된 비디오가 {output_path}에 저장되었습니다.")
        return output_path
    else:
        print("오류: 출력 비디오 파일이 생성되지 않았습니다.")
        return None
