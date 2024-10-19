# Define async function to analyze video and generate text
from vision import process_yolo_results, get_most_common_breed, process_video
from llm import analyze_video, summarize_video, generate_comprehensive_analysis
from config import yolo_model, device, conf_thresh, checkpoint, font_path
from retriever import get_ret, generate_query
import asyncio
import torch
import numpy as np
import torch.nn as nn
from utils import breed_mapping
from config import font_path
import logging
from chat import handle_user_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Model_Echo")

async def analyze_video_and_generate_text(video_path, yolo_model, device, conf_thresh, checkpoint, font_path):
    # YOLO 모델 예측
    results = await asyncio.to_thread(yolo_model.predict, source=video_path, save=True, conf=conf_thresh, stream=True, verbose=False)
    lstm_keypoint_sequence, skeleton_sequence, breed_counter = process_yolo_results(results)

    # 가장 많이 탐지된 품종 선택
    most_common_breed, breed_percentage = get_most_common_breed(breed_counter)
    most_common_breed_ko = breed_mapping.get(most_common_breed, most_common_breed)
    yolo_result = f"가장 많이 탐지된 품종: {most_common_breed_ko} (전체 탐지 중 {breed_percentage:.2f}%)"
    yolo_text = most_common_breed_ko

    print("YOLO 분석 결과:")
    print(yolo_result)
    print(f"yolo_text: {yolo_text}")

    # LSTM 분석
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

    # 모델 파라미터 및 메타데이터 설정
    keypoint_size = checkpoint.get('keypoint_size', 30)
    skeleton_size = checkpoint.get('skeleton_size', 72)
    hidden_size = checkpoint.get('hidden_size', 128)
    num_layers = checkpoint.get('num_layers', 2)
    num_classes = checkpoint.get('num_classes', len(checkpoint.get('all_class_names', [])))
    all_class_names = checkpoint.get('all_class_names', [f'Class_{i}' for i in range(num_classes)])
    metadata = checkpoint.get('metadata', []) 

    # 모델 인스턴스 생성
    lstm_model = ImprovedLSTMModel(keypoint_size, skeleton_size, hidden_size, num_layers, num_classes).to(device)

    # 가중치 직접 로드
    try:
        lstm_model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        print("'model_state_dict' 키를 찾을 수 없습니다. 체크포인트에서 직접 가중치를 로드합니다.")
        model_dict = lstm_model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        lstm_model.load_state_dict(model_dict)

    lstm_model.eval()

    # 슬라이딩 윈도우 함수 정의
    def dynamic_sliding_window(sequence, min_window_size=50, max_window_size=100, min_stride=10):
        sequence_length = len(sequence)
        window_size = min(max_window_size, max(min_window_size, sequence_length // 10))
        stride = max(min_stride, window_size // 2)
        
        windows = []
        for start in range(0, sequence_length - window_size + 1, stride):
            windows.append(sequence[start:start + window_size])
        
        if sequence_length % window_size != 0:
            windows.append(sequence[-window_size:])
        
        return np.array(windows)

    # 슬라이딩 윈도우 적용
    lstm_windows = dynamic_sliding_window(lstm_keypoint_sequence)
    skeleton_windows = dynamic_sliding_window(skeleton_sequence)

    print(f"LSTM 윈도우 형태: {lstm_windows.shape}")
    print(f"스켈레톤 윈도우 형태: {skeleton_windows.shape}")

    # 텐서로 변환
    keypoints_tensor = torch.FloatTensor(lstm_windows).to(device)
    skeleton_tensor = torch.FloatTensor(skeleton_windows).to(device)

    print(f"키포인트 텐서 형태: {keypoints_tensor.shape}")
    print(f"스켈레톤 텐서 형태: {skeleton_tensor.shape}")

    # 키포인트 텐서 형태 조정 (필요한 경우)
    if keypoints_tensor.shape[-2] == 15 and keypoints_tensor.shape[-1] == 2:
        keypoints_tensor = keypoints_tensor.view(keypoints_tensor.shape[0], keypoints_tensor.shape[1], -1)

    print(f"조정된 키포인트 텐서 형태: {keypoints_tensor.shape}")
    print(f"조정된 스켈레톤 텐서 형태: {skeleton_tensor.shape}")

    # LSTM 입력 형태 확인
    assert keypoints_tensor.dim() == 3, f"키포인트 텐서는 3차원이어야 합니다. 현재: {keypoints_tensor.dim()}D"
    assert skeleton_tensor.dim() == 3, f"스켈레톤 텐서는 3차원이어야 합니다. 현재: {skeleton_tensor.dim()}D"
    assert keypoints_tensor.shape[-1] == 30, f"키포인트 텐서의 마지막 차원은 30이어야 합니다. 현재: {keypoints_tensor.shape[-1]}"
    assert skeleton_tensor.shape[-1] == 72, f"스켈레톤 텐서의 마지막 차원은 72여야 합니다. 현재: {skeleton_tensor.shape[-1]}"

    # 예측
    with torch.no_grad():
        outputs = lstm_model(keypoints_tensor, skeleton_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

    # 가장 많이 예측된 클래스 선택
    predicted_class = predicted_classes.mode().values.item()
    overall_probabilities = probabilities.mean(dim=0)

    # 예측 결과와 연관된 메타데이터 가져오기
    predicted_action = all_class_names[predicted_class]
    associated_metadata = metadata[predicted_class] if predicted_class < len(metadata) else {}

    # 예측 결과 출력
    print(f"예측된 클래스: {predicted_action}")
    print(f"클래스별 확률:")
    for i, prob in enumerate(overall_probabilities):
        print(f"  {all_class_names[i]}: {prob.item():.4f}")

    # 메타데이터 출력
    print("\n예측된 클래스의 메타데이터:")
    if associated_metadata:
        print(f"  통증: {associated_metadata.get('pain', 'N/A')}")
        print(f"  질병: {associated_metadata.get('disease', 'N/A')}")
        print(f"  감정: {associated_metadata.get('emotion', 'N/A')}")
        print(f"  비정상 행동: {associated_metadata.get('abnormal_action', 'N/A')}")
    else:
        print("  이 클래스에 대한 메타데이터가 없습니다.")

    # lstm_text 생성
    lstm_text = f"행동: {predicted_action}\n"
    if associated_metadata:
        lstm_text += f"  통증: {associated_metadata.get('pain', 'N/A')}\n"
        lstm_text += f"  질병: {associated_metadata.get('disease', 'N/A')}\n"
        lstm_text += f"  감정: {associated_metadata.get('emotion', 'N/A')}\n"
        lstm_text += f"  비정상 행동: {associated_metadata.get('abnormal_action', 'N/A')}\n"
    else:
        lstm_text += "알 수 없음"

    # 비디오 처리 및 라벨 추가 (선택 사항)
    labeled_video_path = await asyncio.to_thread(process_video, video_path, lstm_text, yolo_text)

    # LLM 분석
    base64Frames, audio_path = await analyze_video(video_path)
    summary_text = await summarize_video(base64Frames, audio_path)

    # Query 생성
    summary_query = generate_query(yolo_text, lstm_text, summary_text)
    print("query :", summary_query)

    # 벡터 검색
    retriever = get_ret()
    try:
        documents = await asyncio.to_thread(retriever.invoke, summary_query)

        print("검색된 문서:")
        if not documents:
            print("검색 결과가 없습니다.")
            rag_text = "검색 결과가 없습니다."
        else:
            # RAG 텍스트로 변환
            rag_text = "\n\n".join([doc.page_content for doc in documents])
            print("RAG Text:")
            print(rag_text)
    except Exception as e:
        print(f"검색 중 오류가 발생했습니다: {str(e)}")
        rag_text = "검색 중 오류가 발생했습니다."

    # 종합 분석 생성
    follow_up_answer = generate_comprehensive_analysis(
        yolo_text=yolo_text,
        lstm_text=lstm_text,
        summary_text=summary_text,
        rag_text=rag_text,
        chat_history=[]  # 초기 분석 결과 포함
    )

    print("Follow Up Answer:")
    print(follow_up_answer)

    # 결과 반환
    return {
        "yolo_text": yolo_text,
        "lstm_text": lstm_text,
        "summary_text": summary_text,
        "rag_text": rag_text,
        "follow_up_answer": follow_up_answer
    }

async def main(video_path):
    # 비디오 분석 수행
    analysis_results = await analyze_video_and_generate_text(video_path, yolo_model, device, conf_thresh, checkpoint, font_path)

    # 사용자 질문 처리
    session_id = "some_session_id"
    user_question = "강아지의 상태를 분석하고 조언해주세요."
    chat_history = []

    response = await handle_user_question(session_id, user_question, chat_history, analysis_results)
    print(response['answer'])

if __name__ == "__main__":
    video_path = "G:/workspace/1006/test/test.mp4"  # 비디오 파일 경로 설정
    asyncio.run(main(video_path))