import os
import requests

from fastapi import FastAPI, Body, HTTPException, UploadFile, File
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

app = FastAPI()
question_store = {}

# RAG 서버 URL
RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://rag_server:8888")

# Ollama 서버 URL
OLLAMA_SERVER_URL = os.getenv("OLLAMA_SERVER_URL", "http://ollama:11434")

# 1번 서버 띄우기
@app.get("/main")
def hello():
    return {"message": "Hello"}

#3. 로그인 하기 위해서 클래스 생성    #basemodel -> type 강제
class LoginUser(BaseModel):
    username: str
    password: str

users = []
users.append(LoginUser(username="1411",password="1411"))
users.append(LoginUser(username="1414",password="1414"))

#2번 로그인
@app.post("/login")
def login(response: Response, user: LoginUser = Body(...)):
    # 로그인 검증
    ok = any(u.username == user.username and u.password == user.password for u in users)
    if not ok:
        return JSONResponse({"ok": False, "reason": "invalid credentials"}, status_code=401)

    # 응답 만들고 쿠키 세팅
    res = JSONResponse({"ok": True})
    res.set_cookie("username", user.username, httponly=True)
    return res

@app.get("/page")
def page(request: Request):
    username = request.cookies.get("username")  # KeyError 방지
    if not username:
        return JSONResponse({"ok": False, "reason": "no cookie"}, status_code=401)
    # username이 등록된 유저인지 확인
    if username in [u.username for u in users]:
        return {"ok": True, "message": f"welcome {username}"}
    return JSONResponse({"ok": False, "reason": "unknown user"}, status_code=403)


def get_current_user(request: Request) -> str:
    username = request.cookies.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다")
    if username not in [u.username for u in users]:
        raise HTTPException(status_code=401, detail="다시 로그인해주세요")
    return username

@app.post("/upload/file")
async def upload_file(request: Request, file: UploadFile = File(...)):
    username = get_current_user(request)

    # 파일을 rag_server로 전달
    files = {
        "file": (
            file.filename,
            await file.read(),
            file.content_type
        )
    }

    rag_res = requests.post(
        f"{RAG_SERVER_URL}/analyze_resume",
        files=files
    )

    if rag_res.status_code != 200:
        raise HTTPException(status_code=500, detail="RAG 서버 파일 분석 실패")

    result = rag_res.json()

    # 사용자별 히스토리 저장
    question_store.setdefault(username, []).append({
        "type": "resume",
        "filename": file.filename,
        "result": result
    })

    return {
        "ok": True,
        "user": username,
        "filename": file.filename,
        "analysis": result
    }

@app.post("/upload")
async def upload(request: Request, text: str = Body(...)):
    username = get_current_user(request)
    rag_res = requests.post(
        f"{RAG_SERVER_URL}/rag/analyze",
        json={"text": text}
    )
    if rag_res.status_code != 200:
        raise HTTPException(status_code=500, detail="RAG 서버 오류")
    result = rag_res.json()
    question_store.setdefault(username, []).append(result)

    return {
        "ok": True,
        "user": username,
        "result": result
    }

@app.get("/history")
def history(request: Request):
    username = get_current_user(request)
    return question_store.get(username, [])

