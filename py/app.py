from models import load_yolo_model, load_lstm_model, process_yolo_results
from utils import process_video
from video_analysis import analyze_video, summarize_video
from retriever import generate_query, get_ret, generate_comprehensive_analysis
from config import YOLO_MODEL_PATH, LSTM_MODEL_PATH, CONF_THRESH, breed_mapping

def main():
    video_path = input("분석할 비디오 파일 경로를 입력하세요: ")
    if not os.path.exists(video_path):
        print("오류: 비디오 파일을 찾을 수 없습니다.")
        return
    
    yolo_model = load_yolo_model(YOLO_MODEL_PATH)
    lstm_model, all_class_names, metadata = load_lstm_model(LSTM_MODEL_PATH)
    
    results = yolo_model.predict(source=video_path, save=True, conf=CONF_THRESH, stream=True, verbose=False)
    lstm_keypoint_sequence, skeleton_sequence, breed_counter = process_yolo_results(results)
    most_common_breed, breed_percentage = get_most_common_breed(breed_counter)

    # 영어 품종 이름을 한국어로 변환
    most_common_breed_ko = breed_mapping.get(most_common_breed, most_common_breed)
    yolo_result = f"가장 많이 탐지된 품종: {most_common_breed_ko} (전체 탐지 중 {breed_percentage:.2f}%)"
    yolo_text = most_common_breed_ko

    output_video_path = process_video(video_path, lstm_text, yolo_text)

    if output_video_path:
        print(f"비디오 경로: {output_video_path}")
    else:
        print("비디오 생성에 실패했습니다.")

    base64Frames, audio_path = analyze_video(video_path)
    summary_text = summarize_video(base64Frames, audio_path)

    retriever = get_ret()
    summary_query = generate_query(yolo_text, lstm_text, summary_text)
    print("생성된 쿼리:", summary_query)

    try:
        documents = retriever.get_relevant_documents(summary_query)[:2]
        rag_text = "\n\n".join([doc.page_content for doc in documents]) if documents else ""
    except Exception as e:
        print(f"검색 중 오류가 발생했습니다: {str(e)}")
        rag_text = ""

    response = generate_comprehensive_analysis(yolo_text, lstm_text, summary_text, rag_text)
    print("종합 분석 결과:")
    print(response)

    chat_history = [response]

    while True:
        user_question = input("답변에 만족하셨나요? 추가로 궁금하신 점이 있으시면 질문해 주세요 (종료:exit) : ")
        if user_question.lower() == 'exit':
            break

        chat_history.append({"role": "user", "content": user_question})

        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "당신은 강아지 행동 분석 전문가입니다."},
                    *chat_history
                ],
                max_tokens=1000
            )

            ai_response = response.choices[0].message.content
            chat_history.append({"role": "assistant", "content": ai_response})
            print("응답:\n", ai_response)
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            print("응답을 생성하는 데 문제가 발생했습니다. 다시 시도해 주세요.")

if __name__ == "__main__":
    main()