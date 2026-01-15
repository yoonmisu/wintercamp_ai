from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import pymupdf4llm
import tiktoken
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings
import requests
import json

app = FastAPI()

class UploadRequest(BaseModel):
    full_text: str
    chunk_size: int

# 전역 변수로 모델과 클라이언트 초기화
embedding_model = None
chroma_client = None

@app.on_event("startup")
async def startup_event():
    global embedding_model, chroma_client
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    print("Initializing ChromaDB client...")
    chroma_client = chromadb.PersistentClient()
    print("Startup complete!")

def split_text(full_text, chunk_size):
    encoder = tiktoken.encoding_for_model("gpt-5")
    total_encoding = encoder.encode(full_text)
    total_token_count = len(total_encoding)
    text_list = []
    for i in range(0, total_token_count, chunk_size):
        chunk = total_encoding[i: i+chunk_size]
        decoded = encoder.decode(chunk)
        text_list.append(decoded)

    return text_list

@app.post("/upload")
def upload(request: UploadRequest):
    global embedding_model, chroma_client

    chunk_list = split_text(request.full_text, 1000)

    embeddings = embedding_model.encode(chunk_list)
    print(embeddings)

    class MyEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            return embedding_model.encode(input).tolist()

    collection_name = 'samsung_collection6'

    # 기존 collection이 있으면 삭제
    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass

    samsung_collection = chroma_client.create_collection(name=collection_name, embedding_function=MyEmbeddingFunction())

    # 문서마다 id 만들
    id_list = []
    for index in range(len(chunk_list)):
        id_list.append(f'{index}')

    samsung_collection.add(documents=chunk_list, ids=id_list)

    return {"ok": True, "chunks": len(chunk_list)}



class QueryRequest(BaseModel):
    query: str

@app.post("/answer")
def llm_response(request: QueryRequest):
    global embedding_model, chroma_client

    collection_name = 'samsung_collection6'

    class MyEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            return embedding_model.encode(input).tolist()

    # 기존 컬렉션 가져오기
    samsung_collection = chroma_client.get_collection(name=collection_name, embedding_function=MyEmbeddingFunction())

    # 사용자 질문으로 refer 검색
    retrieved_doc = samsung_collection.query(query_texts=request.query, n_results=3)
    refer = retrieved_doc['documents'][0]

    # Ollama LLM 호출
    url = "http://ollama:11434/api/generate"

    payload = {
        "model": "gemma3:1b",
        "prompt": f'''You are a business analysis expert in Korea.
                    Please find answers to users' questions in our *Context*. If not, please direct them to the company.
                    Please organize your answers so users can understand them.
            *Context*:
            {refer}
            *Question*: {request.query}

            Answer in Korean:''',
        "stream": False
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    return {"response": response.json()["response"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)