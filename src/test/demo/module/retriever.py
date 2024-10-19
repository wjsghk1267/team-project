from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
import joblib
import numpy as np
from config import db_client, model, client
from llm import generate_query


def generate_query(yolo_text, lstm_text, summary_text):
    # 쿼리 템플릿에 통합된 텍스트를 삽입
    query_template = f"""
    당신은 강아지 행동 분석 전문가입니다. 주어진 정보를 바탕으로 강아지의 상태를 종합적으로 요약해야 합니다.

    ** 입력 정보 **
    1. YOLO 모델 결과: {yolo_text} (강아지 품종 정보)
    2. LSTM 모델 결과: {lstm_text} (강아지 행동, 감정, 통증여부, 질병여부, 비정상 행동 정보)
    3. LLM 비디오/오디오 분석 요약: {summary_text} (강아지의 시각적 행동 패턴, 소리, 환경 정보 등)

    ** 요약 지침 **
    1. 모든 입력 정보를 통합하여 강아지의 상태를 종합적으로 설명하세요.
    2. 품종, 주요행동, 감정상태, 건강상태, 환경적 요인을 포함해주세요.
    3. 영상 속 강아지의 특이사항이나 문제점, 솔루션이 필요하다면 말해주세요.
    4. 전체 요약은 2-3문장으로 제한하세요.
    5. 요약은 키워드 중심으로 작성하고, 불필요한 관사나 조사는 생략하세요.

    위의 지침을 참고하여, 주어진 모든 정보를 바탕으로 강아지의 상태를 종합적으로 요약해주세요.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": query_template},
            {"role": "user", "content": "비디오를 종합적으로 요약해주세요."}
        ],
        max_tokens=100
    )

    query = response.choices[0].message.content
    return query

def get_ret():
    # 임베딩 모델 설정
    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # 저장된 TfidfVectorizer 로드
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    
    # 커스텀 임베딩 클래스 정의
    class CustomEmbeddings:
        def __init__(self, embeddings_model, vectorizer):
            self.embeddings_model = embeddings_model
            self.vectorizer = vectorizer
        
        def embed_query(self, text):
            dense_embedding = self.embeddings_model.embed_query(text)
            sparse_embedding = self.vectorizer.transform([text]).toarray()[0]
            combined_embedding = np.concatenate([dense_embedding, sparse_embedding])
            if combined_embedding.shape[0] > 2048:
                combined_embedding = combined_embedding[:2048]
            elif combined_embedding.shape[0] < 2048:
                padding = np.zeros(2048 - combined_embedding.shape[0])
                combined_embedding = np.concatenate([combined_embedding, padding])
            return combined_embedding.tolist()
    
    custom_embeddings = CustomEmbeddings(embeddings_model, vectorizer)
    
    # MongoDB 컬렉션 설정
    dbName_json = "db"
    collectionName_json = "dog_document"
    collection_json = db_client[dbName_json][collectionName_json]

    vectorStore_json = MongoDBAtlasVectorSearch(
        embedding=custom_embeddings,
        collection=collection_json,
        index_name='dog_index2',
        embedding_key="embedding",
        text_key="content"
    )
    # 검색기 생성
    retriever_json = vectorStore_json.as_retriever()
    return retriever_json