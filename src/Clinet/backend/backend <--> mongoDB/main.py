# main.py

import os
import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
import uvicorn
from ultralytics import YOLO
import torch
import logging

# model_echo.py에서 필요한 함수 임포트
from Model_Echo import analyze_video_and_generate_text

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI()

# MongoDB 연결 설정
MONGODB_URL = "mongodb+srv://yoonsun2596:qwer1234@tm2.hl7a3.mongodb.net"
client = AsyncIOMotorClient(MONGODB_URL)
db = client.get_database("LOG")
collection = db.get_collection("chat_log")

# 모델 실행을 위한 설정 (필요에 따라 수정)
YOLO_MODEL = YOLO(r"D:\kdt_240424\workspace\project3\AI_models\flutter\Animal_KETPOINT\yolov8m-pose_100_epochs\weights\best.pt")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONF_THRESH = 0.5  # 신뢰도 임계값
CHECKPOINT = torch.load(
    r'D:\kdt_240424\workspace\project3\AI_models\flutter\Animal_KETPOINT\yolov8m-pose_100_epochs\weights\lstm_model_1010.pt',
    map_location=DEVICE
)
FONT_PATH = r"D:\kdt_240424\workspace\project3\AI_models\flutter\malgun.ttf"  # 한글 폰트 파일 경로

# 분석 결과를 저장할 Pydantic 모델
class AnalysisResult(BaseModel):
    yolo_text: str
    lstm_text: str
    summary_text: str
    rag_text: str
    follow_up_answer: str
    upload_time: datetime.datetime

@app.post("/upload/", response_model=AnalysisResult)
async def upload_video(file: UploadFile = File(...)):
    import shutil  # 파일 저장을 위한 임포트 추가

    try:
        # 파일 저장
        file_location = f"uploaded_videos/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"파일 저장 완료: {file_location}")

        # AI 분석 수행
        analysis_results = await analyze_video_and_generate_text(
            video_path=file_location,
            yolo_model=YOLO_MODEL,
            device=DEVICE,
            conf_thresh=CONF_THRESH,
            checkpoint=CHECKPOINT,
            font_path=FONT_PATH
        )

        logger.info("AI 분석 완료")

        # MongoDB에 저장할 데이터 구조
        document = {
            "filename": file.filename,
            "upload_time": datetime.datetime.utcnow(),
            "yolo_text": analysis_results.get("yolo_text"),
            "lstm_text": analysis_results.get("lstm_text"),
            "summary_text": analysis_results.get("summary_text"),
            "rag_text": analysis_results.get("rag_text"),
            "follow_up_answer": analysis_results.get("follow_up_answer")
        }

        # MongoDB에 삽입
        result = await collection.insert_one(document)

        logger.info(f"MongoDB에 데이터 저장 완료: {result.inserted_id}")

        # 응답 생성
        return AnalysisResult(
            yolo_text=document["yolo_text"],
            lstm_text=document["lstm_text"],
            summary_text=document["summary_text"],
            rag_text=document["rag_text"],
            follow_up_answer=document["follow_up_answer"],
            upload_time=document["upload_time"]
        )

    except Exception as e:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")

# http://localhost:8000
@app.get("/")
def read_root():
    return {"message": "Hello World"}

# http://localhost:8000/docs
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
