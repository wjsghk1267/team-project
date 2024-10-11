from pymongo import MongoClient
from prompt import comprehensive_analysis_template

def generate_query(yolo_text, lstm_text, summary_text):
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
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": query_template},
            {"role": "user", "content": "비디오를 종합적으로 요약해주세요."}
        ],
        max_tokens=100
    )

    query = response.choices[0].message.content
    return query

def get_ret():
    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    index_name_json = 'dog_test'
    dbName_json = "dbsparta"
    collectionName_json = "dog"generate_comprehensive_analysis
    collection_json = db_client[dbName_json][collectionName_json]

    vectorStore_json = MongoDBAtlasVectorSearch(
        embedding=embeddings_model,
        collection=collection_json,
        index_name=index_name_json,
        embedding_key="embedding",
        text_key="content"
    )
    retriever_json = vectorStore_json.as_retriever()
    return retriever_json

def generate_comprehensive_analysis(yolo_text, lstm_text, summary_text, rag_text):
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18", 
        messages=[
            {"role": "system", "content": comprehensive_analysis_template.format(
                yolo_text=yolo_text,
                lstm_text=lstm_text,
                summary_text=summary_text,
                rag_text=rag_text
            )},
            {"role": "user", "content": "강아지의 상태를 분석하고 조언해주세요."}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content