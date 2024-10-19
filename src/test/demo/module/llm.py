
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import client
import asyncio
import os
import cv2
import base64
import io
from moviepy.editor import VideoFileClip
from PIL import Image
import logging
from config import model

logger = logging.getLogger(__name__)

chat_analysis_template = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 20년 경력의 수의사이자 강아지 행동 분석 전문가입니다. 주어진 모든 정보를 종합하여 강아지의 상태를 분석하고 전문적인 조언을 제공해야 합니다.

    ** 입력 정보 ** 
    1. YOLO 분석 결과: {yolo_text}
    2. LSTM 분석 결과: {lstm_text}
    3. LLM 비디오 분석 결과: {summary_text}
    4. RAG 문서 정보: {rag_text}

    ** 유저와의 상호작용 ** 
    모든 대화는 한국어로 진행됩니다.

    ** 분석 및 답변 지침 ** 
    1. 모든 입력 정보를 종합하여 강아지의 상태, 행동, 감정, 건강 상태를 정확히 파악하세요.
    2. YOLO 결과로부터 강아지의 품종과 외형적 특징을 고려하세요.
    3. LSTM 결과를 바탕으로 강아지의 행동 패턴과 감정 상태, 통증여부, 이상행동 여부를 분석하세요.
    4. LLM 비디오 분석 결과를 통해 전반적인 상황 맥락을 이해하세요.
    5. RAG 문서 정보를 활용하여 관련된 전문 지식을 답변에 통합하세요.
    6. 문제가 있다면 그 원인을 간단히 설명하고, 구체적이고 실행 가능한 해결책을 제시하세요.
    7. 보호자가 즉시 실천할 수 있는 실용적인 조언을 포함하세요.
    8. 필요한 경우 전문가 상담이나 병원 방문을 권유하세요.
    9. 답변은 전문적이면서도 이해하기 쉽게 작성하세요.
    
    ** 답변 구조 ** 
    유저의 질문에 대해 주어진 모든 정보를 바탕으로 한 문단으로 종합적이고 직접적인 답변을 작성하세요.

    주어진 모든 정보를 종합하여 전문적이고 실용적인 분석과 조언을 제공해주세요.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

comprehensive_analysis_template = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 20년 경력의 수의사이자 강아지 행동 분석 전문가입니다. 주어진 모든 정보를 종합하여 강아지의 상태를 분석하고 전문적인 조언을 제공해야 합니다.

    ** 입력 정보 **
    1. YOLO 분석 결과: {yolo_text}
    2. LSTM 분석 결과: {lstm_text}
    3. LLM 비디오 분석 결과: {summary_text}
    4. RAG 문서 정보: {rag_text}

    ** 유저와의 상호작용 **
    모든 대화는 한국어로 진행됩니다.

    ** 분석 및 답변 지침 **
    1. 모든 입력 정보를 종합하여 강아지의 상태, 행동, 감정, 건강 상태를 정확히 파악하세요.
    2. YOLO 결과로부터 강아지의 품종과 외형적 특징을 고려하세요.
    3. LSTM 결과를 바탕으로 강아지의 행동 패턴과 감정 상태, 통증여부, 이상행동 여부를 분석하세요.
    4. LLM 비디오 분석 결과를 통해 전반적인 상황 맥락을 이해하세요.
    5. RAG 문서 정보를 활용하여 관련된 전문 지식을 답변에 통합하세요.
    6. 문제가 있다면 그 원인을 간단히 설명하고, 구체적이고 실행 가능한 해결책을 제시하세요.
    7. 보호자가 즉시 실천할 수 있는 실용적인 조언을 포함하세요.
    8. 필요한 경우 전문가 상담이나 병원 방문을 권유하세요.
    9. 답변은 전문적이면서도 이해하기 쉽게 작성하세요.
    
    ** 답변 구조 **
    1. 종합적 상황 요약 (YOLO, LSTM, LLM 결과 통합)
    2. 원인 분석 (해당되는 경우)
    3. 맞춤형 해결책 또는 권장 행동 (RAG 정보 활용)
    4. 추가 조언 또는 주의사항 (필요한 경우)
    5. 결론 및 격려의 말

    주어진 모든 정보를 종합하여 전문적이고 실용적인 분석과 조언을 제공해주세요.
    """),
    MessagesPlaceholder(variable_name="chat_history")
])

# LLM 분석 템플릿
llm_analysis_template = """
    ** 역할 **
    당신은 20년간 강아지에 대해 공부한 수의사 및 행동 분석가입니다.
    강아지에 대한 풍부한 경험과 전문적인 지식을 보유하고 있습니다.
    제공받은 영상과 오디오를 확인하여 강아지에 대한 분석을 진행합니다.

    ** 분석 프로세스 **
    1. 영상 분석:
       - 제공된 프레임 단위 이미지를 순서대로 분석합니다.
       - 강아지의 자세, 움직임, 표정을 관찰합니다.
       - 주변 환경과 상황적 맥락을 파악합니다.
       - 보호자가 있다면 그들의 행동도 분석합니다.

    2. 오디오 분석:
       - 오디오 파형과 주파수 분석으로 강아지의 소리(짖음, 울음 등)의 특성을 파악합니다.
       - 배경 소음이나 다른 소리들도 고려합니다.

    3. 종합 분석:
       - 영상과 오디오 정보를 종합하여 강아지의 전반적인 상태를 평가합니다.
       - 행동 패턴, 감정 상태, 건강 상태에 대한 의견을 제시합니다.
       - 특이사항이나 문제점이 있다면 언급합니다.

    ** 분석 결과 형식 **
    분석 결과를 다음 형식으로 제공해 주세요:
    1. 영상 분석 요약: [영상에서 관찰된 주요 사항들을 간결하게 서술]
    2. 오디오 분석 요약: [오디오에서 파악된 주요 정보를 간결하게 서술]
    3. 종합 분석: [영상과 오디오 정보를 종합한 전반적인 분석 결과]

    주어진 정보를 바탕으로 강아지의 상태를 전문가적 관점에서 분석해 주세요.
"""

async def analyze_video(file_path, seconds_per_frame=2):
    base64Frames = []
    base_video_file, _ = os.path.splitext(file_path)
    video = cv2.VideoCapture(file_path)
    if not video.isOpened():
        raise Exception("Error opening video file")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)

    curr_frame = 0
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break

        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip

    video.release()

    clip = VideoFileClip(file_path)
    audio_path = f"{base_video_file}.mp3"
    try:
        if clip.audio:
            await asyncio.to_thread(clip.audio.write_audiofile, audio_path, bitrate="32k")
            clip.audio.close()
        else:
            audio_path = None
    except Exception as e:
        audio_path = None
    clip.close()

    return base64Frames, audio_path

# Define function to summarize video
async def summarize_video(base64Frames, audio_path):
    summary_text = ""

    # 비디오 프레임을 표시합니다.
    if base64Frames:  # base64Frames가 비어있지 않은지 확인
        for img in base64Frames:
            # base64로 인코딩된 이미지를 디코딩하여 PIL 이미지로 변환
            image_data = base64.b64decode(img)
            image = Image.open(io.BytesIO(image_data))  # BytesIO를 사용하여 이미지 열기
            
    # 오디오 경로가 None이 아닌 경우에만 transcription을 실행합니다.
    if audio_path is not None:
        with open(audio_path, 'rb') as audio_file:
            transcription = await asyncio.to_thread(client.audio.transcriptions.create,
                                                    model="whisper-1",
                                                    file=audio_file)
            audio_text = transcription.text

            print("Transcription 완료:", audio_text, '\n')
            summary_text += audio_text + "\n"

            response_both = await asyncio.to_thread(client.chat.completions.create,
                                                    model=model,
                                                    messages=[
                                                        {"role": "system", "content": llm_analysis_template},
                                                        {"role": "user", "content": [
                                                            "이건 비디오 영상의 프레임 이미지.",
                                                            *map(lambda x: {"type": "image_url",
                                                                            "image_url": {"url": f"data:image/jpg;base64,{x}", "detail":"low"}}, base64Frames),
                                                            {"type": "text", "text": f"이건 비디오 영상의 오디오 {audio_text}"}
                                                        ]},
                                                    ],
                                                    temperature=0.6)
            print(response_both.choices[0].message.content)
            summary_text += response_both.choices[0].message.content + "\n"
            print("\n", "="*100, "\n")

    else:
        print("오디오 내용이 없습니다. Transcription은 스킵합니다", '\n')

        # 오디오가 없는 경우에도 비디오 프레임에 대한 분석을 수행합니다.
        response_vis = await asyncio.to_thread(client.chat.completions.create,
                                               model=model,
                                               messages=[
                                                   {"role": "system", "content": llm_analysis_template},
                                                   {"role": "user", "content": [
                                                       "이건 비디오 영상의 프레임 이미지.",
                                                       *map(lambda x: {"type": "image_url",
                                                                       "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames)
                                                   ]},
                                               ],
                                               temperature=0.6)
        print(response_vis.choices[0].message.content)
        summary_text += response_vis.choices[0].message.content + "\n"
        print("\n", "="*100, "\n")

    return summary_text

def chat_analysis(yolo_text, lstm_text, summary_text, rag_text, user_question, chat_history):
    try:
        # Creating the prompt message
        prompt_message = chat_analysis_template.format(
            yolo_text=yolo_text,
            lstm_text=lstm_text,
            summary_text=summary_text,
            rag_text=rag_text,
            question=user_question,
            chat_history=chat_history
        )
        
        # Create the messages list
        messages = [
            {"role": "system", "content": prompt_message},
            {"role": "user", "content": user_question}
        ]
        
        # Request completion from the API
        response = client.chat.completions.create(
            model=model , 
            messages=messages,
            max_tokens=1000,
            temperature=0.6
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"generate_comprehensive_analysis 중 오류 발생: {str(e)}")
        return ""

# 초기 분석 결과
def generate_comprehensive_analysis(yolo_text, lstm_text, summary_text, rag_text, chat_history):
    try: 
        response = client.chat.completions.create(
            model=model , 
            messages=[
                {"role": "system", "content": comprehensive_analysis_template.format(
                    yolo_text=yolo_text,
                    lstm_text=lstm_text,
                    summary_text=summary_text,
                    rag_text=rag_text,
                    # question=question,
                    chat_history=chat_history
                )},
                {"role": "user", "content": "강아지의 상태를 분석하고 조언해주세요."}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"generate_comprehensive_analysis 중 오류 발생: {str(e)}")
        return ""
