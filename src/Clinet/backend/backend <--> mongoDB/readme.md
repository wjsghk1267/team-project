백엔드 FastAPI 와 몽고DB를 연결 시키는 테스트 코드
작동 확인 완료

가상환경 실행 후 python main.py를 실행하면 서버 구축
구축 후 하단에 FastAPI에서 제공하는 테스트 서버에서 동영상 선택 후 execute 버튼을 누르면 Model_Echo.py 결과물인 텍스트를 자동으로 연결 되어 있는 MongoDB에 저장

몽고 DB에 저장 된 결과 확인 완료
![image](https://github.com/user-attachments/assets/2776fded-594a-4083-ab54-c4ed93159f82)

http://localhost:8000 (서버 확인)
http://localhost:8000/docs (동영상 업로드)
