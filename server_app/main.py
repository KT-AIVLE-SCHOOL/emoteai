from preprocess import pipeline
import os
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()
path = os.getcwd()

@app.post("/api/uploadAudio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # 저장할 디렉토리 설정
        UPLOAD_DIRECTORY = "./uploaded_audio"
        if not os.path.exists(UPLOAD_DIRECTORY):
            os.makedirs(UPLOAD_DIRECTORY)
        
        file_path = os.path.join(UPLOAD_DIRECTORY, f"test.wav")
        
        # # 파일 경로 설정
        # file_path = os.path.join(UPLOAD_DIRECTORY, f"{file.filename}.wav")
        
        # # 파일 저장
        # with open(file_path, "wb") as buffer:
        #     content = await file.read()
        #     buffer.write(content)
        
        return JSONResponse(content={"success": True, "message": str(pipeline(file_path))}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)