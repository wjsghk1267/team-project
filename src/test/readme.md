# AI 파이프라인 정리

1. **YOLO_v8 모델 분석**: 비디오(강아지 품종, keypoint, bounding box)

2. **LSTM 모델 분석**: 키포인트(강아지 행동패턴, 감정, 통증여부, 비정상 행동 정보)

3. **LLM 모델 분석**: 비디오(영상, 오디오 분석 - 요약)

4. **Retriever 생성**: 모델 분석 결과(yolo, lstm, llm 텍스트) - LLM 요약

5. **Vector Search, RAG Context 생성**

6. **LLM 모델 최종 답변 생성**: 
   - 품종
   - 행동패턴
   - 감정
   - 통증여부
   - 비정상 행동 정보
   - LLM 요약
   - RAG 문서 정보
