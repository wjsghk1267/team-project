# Work List

## A. 프롬프트 → LLM
- **문제행동 솔루션**: 응답 조건 [반려동물 정보{나이, 성별, 품종, 성격 등}에 맞는 대사 생성 = 글자수 제한(10자)]
- **감정, 통증, 행동 분석**: 1번 옵션 + (분석정보{감정, 통증 여부, 이상 행동})에 맞는 대사 생성.

## B. LLM → 대사생성
- 데이터 JSON 타입으로 저장

### Example

```json
{
  "pets": [
    {
      "id": "pet1",
      "name": "Buddy",
      "age": "3",
      "gender": "Male",
      "breed": "Labrador",
      "personality": "Playful",
      "utterances": [
        {
          "context": "problem_behavior",
          "response": "저 너무 지루해요! 더 놀아주세요!",
          "character_limit": 10
        },
        {
          "context": "emotion_pain_behavior",
          "emotion": "happy",
          "pain": "no",
          "behavior": "playful",
          "response": "오늘 너무 신나요! 함께 놀아요!",
          "character_limit": 10
        }
      ]
    },
    {
      "id": "pet2",
      "name": "Mittens",
      "age": "7",
      "gender": "Female",
      "breed": "Siamese",
      "personality": "Quiet",
      "utterances": [
        {
          "context": "problem_behavior",
          "response": "혼자 있는 게 좋네요. 조금만 기다려 주세요.",
          "character_limit": 10
        },
        {
          "context": "emotion_pain_behavior",
          "emotion": "sad",
          "pain": "yes",
          "behavior": "inactive",
          "response": "요즘 좀 아파요. 조용히 쉬고 싶어요.",
          "character_limit": 10
        }
      ]
    }
  ]
}
