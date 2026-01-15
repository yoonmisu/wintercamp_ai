import os

import fitz
import pymupdf4llm
import requests
from fastapi import FastAPI, Body, UploadFile, File
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response, RedirectResponse, JSONResponse
from fastapi import HTTPException

# from backend import rag_server
# from backend.rag_server import llm_response

app = FastAPI()

# RAG 서버 URL
RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://rag_server:8888")

# Ollama 서버 URL
OLLAMA_SERVER_URL = os.getenv("OLLAMA_SERVER_URL", "http://ollama:11434")

full_text = ""
RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://localhost:7777")

# 1번 서버 띄우기
@app.get("/hello")
def hello():
    return {"message": "Hello World"}

#3. 로그인 하기 위해서 클래스 생성    #basemodel -> type 강제
class LoginUser(BaseModel):
    username: str
    password: str

users = []
users.append(LoginUser(username="park",password="q1w2e3"))
users.append(LoginUser(username="choi",password="q1w2e3"))

#2번 로그인
@app.post("/login")
def login(response: Response, user: LoginUser = Body()): #option enter  response ->sh,  body ->
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


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    global full_text
    # 1) 쿠키 사용자 인증
    username = get_current_user(request)

    # 2) PDF만 허용 (content-type은 클라이언트가 속일 수 있어서 확장자도 같이 체크)
    filename = (file.filename or "").lower()
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다")

    if file.content_type not in (None, "application/pdf"):
        # 일부 클라이언트는 content_type이 None인 경우도 있어 None은 허용
        raise HTTPException(status_code=400, detail="content-type이 PDF가 아닙니다")

    # 3) 저장 없이 메모리에서 바이트로 읽기
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="빈 파일입니다")

    # 4) PyMuPDF로 바이트 스트림 열고 → pymupdf4llm로 Markdown 텍스트 추출
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = pymupdf4llm.to_markdown(doc)  # 문서 전체를 Markdown 텍스트로
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF 파싱 실패: {e}")
    finally:
        try:
            doc.close()
        except Exception:
            pass

#사용자 질문 -> 벡터디비에 질의 -> 결과값을 refer에 저장 -> LLM에 연결해서 답변
def upload_to_rag(full_text, chunk_size: int = 1024):
    # rag 서버에 연결
    response = requests.post(
        f"{RAG_SERVER_URL}/upload",
        json={"full_text": full_text, "chunk_size": chunk_size},\
        timeout=60,
    )
    return response.json()

def lim_response(query):
    response = requests.post(
        f"{RAG_SERVER_URL}/answer",
        json={"query": query},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()

@app.post("/query")
def query(request: Request, query = Body()):
    username = get_current_user(request)
    upload_to_rag(full_text,500)
    lr = lim_response(query)
    return{
        "ok": True,
        "user": username,
        "query": query,
        "lim_response": lr
    }

    # 여기서 full_text 변수를 원하는 대로 후처리/저장(DB 등)하면 됨
    # 지금은 예시로 일부 정보만 반환
    return {
        "ok": True,
        "user": username,
        "chars": len(full_text),
        "preview": full_text[:500],  # 너무 길면 잘라서
    }