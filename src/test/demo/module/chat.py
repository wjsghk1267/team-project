from typing import List, Dict, Any
from llm import chat_analysis
import logging
from retriever import get_ret

logger = logging.getLogger(__name__)

async def handle_user_question(session_id: str, user_question: str, chat_history: List[Dict[str, Any]], analysis_results: Dict[str, Any]):
    try:
        # 분석 결과 추출
        yolo_text = analysis_results.get("yolo_text", "")
        lstm_text = analysis_results.get("lstm_text", "")
        summary_text = analysis_results.get("summary_text", "")
        rag_text = analysis_results.get("rag_text", "")

        # 새로운 질문으로 래그 
        # 벡터 검색
        retriever = get_ret()
        try:
            documents = retriever.invoke(user_question)

            logger.info("검색된 문서:")
            if not documents:
                logger.info("검색 결과가 없습니다.")
                rag_text = "검색 결과가 없습니다."
            else:
                # RAG 텍스트로 변환
                rag_text = "\n\n".join([doc.page_content for doc in documents])
                logger.info("RAG Text:")
                logger.info(rag_text)
        except Exception as e:
            logger.error(f"검색 중 오류가 발생했습니다: {str(e)}")
            rag_text = "검색 중 오류가 발생했습니다."

        # 사용자 질문을 chat_history에 추가
        chat_history.append({"role": "user", "content": user_question})

        # 분석 결과 생성
        follow_up_answer = chat_analysis(
            yolo_text=yolo_text,
            lstm_text=lstm_text,
            summary_text=summary_text,
            rag_text=rag_text,
            user_question=user_question, 
            chat_history=chat_history
        )

        # 대화 내역에 응답 추가
        chat_history.append({"role": "assistant", "content": follow_up_answer})

        # 결과 반환
        # print(f'종합 답변 : {follow_up_answer}')
        # return follow_up_answer
        # 결과 반환 
        # print(f'종합 답변 : {follow_up_answer}')
        return {
            'answer': follow_up_answer,
        }
    except Exception as e:
        logger.error(f"handle_user_question 중 오류 발생: {str(e)}")
        return "죄송합니다. 질문을 처리하는 중 오류가 발생했습니다."
    
