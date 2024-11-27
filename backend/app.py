# 전체 시스템 업데이트: JD 키워드를 추출하고 FP 점수를 계산하여 벡터 스토어에 반영하는 방식으로 기존 코드를 수정합니다.

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from utils import save_uploaded_file, load_multiple_pdfs, extract_metadata_llm, input_metadata, split_and_chunk_documents, Embedding, generate_answer_with_llm, backup_processed_pdfs, calculate_fp_scores, calculate_tp_scores, extract_jd_metadata_llm
from retrieval import retrieve_candidates
from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str
from models import embeddings_model
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import asyncio
import logging
import os
import pickle
from fastapi.encoders import jsonable_encoder  # 추가

app = FastAPI()

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 동시성 관리를 위한 Lock
lock = asyncio.Lock()

app.mount(
    "/static",
    StaticFiles(directory=r'C:\Users\edu49\myEnv\Scripts\HR_Project\static'),
    name="static",
)

# 전역 변수로 벡터스토어 선언
vectorstore = Chroma(
    collection_name="hr_list",
    persist_directory="./chroma_db",
    embedding_function=embeddings_model  # 수정된 부분
)
# 애플리케이션 시작 시 기존 벡터스토어 로드
persist_directory = "./chroma_db"

if os.path.exists(persist_directory):
    embedding_function = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    logging.info("Existing vectorstore loaded successfully.")
else:
    logging.info("No existing vectorstore found. It will be created upon first upload.")

# 루트 엔드포인트에서 index.html 반환
@app.get("/", summary="Root Endpoint", description="Basic root endpoint", response_class=HTMLResponse)
async def root():
    with open("frontend/index.html", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# PDF 파일 업로드 엔드포인트
@app.post("/upload/", summary="Upload a PDF file", description="Uploads a PDF file to the server.")
async def upload_pdf(file: UploadFile = File(...), request: PromptRequest = None):
    global vectorstore  # 전역 벡터스토어 사용
    logging.info(f"Received file upload request: {file.filename}")

    async with lock:
        if not file.filename.endswith(".pdf"):
            return JSONResponse(content={"error": "Only PDF files are allowed."}, status_code=400)

        # 파일 저장
        file_location = save_uploaded_file(file, "pdf_data")
        logging.info(f"File saved at {file_location}")
        
         # 기존 PDF 데이터 확인
        pdf_docs_path = "pdf_data/pdf_docs.pkl"
        if os.path.exists(pdf_docs_path):
            with open(pdf_docs_path, "rb") as f:
                existing_pdf_docs = pickle.load(f)
        else:
            existing_pdf_docs = []

        # PDF 데이터 로드
        all_data, all_text = load_multiple_pdfs("pdf_data")  # all_data -> PDF 직접 로드하여 각 페이지에 리스트
        
        if not all_data:
            logging.error("No text extracted from the uploaded PDF files.")
            return JSONResponse(content={"error": "No text could be extracted from the uploaded PDF."}, status_code=404)

        # 메타데이터 추출
        responses, resumes = extract_metadata_llm(all_text)
        pdf_docs = input_metadata(responses, resumes, all_data)


        # 중복된 문서 필터링
        unique_pdf_docs = []
        existing_contents = [doc.page_content for doc in existing_pdf_docs]

        for doc in pdf_docs:
            if doc.page_content not in existing_contents:
                unique_pdf_docs.append(doc)

        # 기존 데이터에 새 데이터를 추가하고 파일로 저장
        if unique_pdf_docs:
            updated_pdf_docs = existing_pdf_docs + unique_pdf_docs
            with open(pdf_docs_path, "wb") as f:
                pickle.dump(updated_pdf_docs, f)
            logging.info(f"pdf_docs updated and saved to {pdf_docs_path}")
            
        # 메타데이터 추출 및 페이지 분리, 청크화 및 토큰화
        split_docs = split_and_chunk_documents(pdf_docs)

        # 벡터스토어에 저장 (업데이트)
        vectorstore = Embedding(split_docs, vectorstore)
        
        # pdf_data의 모든 파일을 pdf_data_backup으로 백업
        backup_processed_pdfs('pdf_data', backup_directory="pdf_data_backup")

        logging.info(f"Total number of Tokens in vectorstore: {len(split_docs)}")

        return {"info": f"file '{file.filename}' saved and processed successfully"}


# 쿼리 기반으로 추천 답변을 생성하는 엔드포인트
@app.post("/retrieve_and_answer/", summary="Retrieve and answer based on prompt", description="Retrieves candidates and generates an answer based on the given prompt.")
async def retrieve_and_answer(request: PromptRequest):
    global vectorstore

    prompt = request.prompt

    if vectorstore is None:
        logging.error("Vectorstore is not initialized.")
        return JSONResponse(content={"answer": "Vector store is not initialized. Please upload a PDF first."}, status_code=400)

    # JD 관련 키워드 추출 및 점수 계산 (FP 계산)
    jd_keywords =  extract_jd_metadata_llm(prompt)
    

    retrieved_docs = retrieve_candidates(prompt, vectorstore=vectorstore)
    
    if not retrieved_docs:
        logging.warning("No documents retrieved for the prompt.")
        return JSONResponse(content={"answer": "No relevant documents found for the provided prompt."})

    
    

    
    logging.info(f"Number of documents retrieved: {len(retrieved_docs)}")
    for i, doc in enumerate(retrieved_docs, 1):
        logging.info(f"Retrieved Document {i}: {doc.page_content[:100]}...")  # 첫 100자만 출력

    # pkl 파일에서 person_korean_name이 일치하는 문서 찾기
    pdf_docs_path = "pdf_data/pdf_docs.pkl"
    matched_docs = []
    if os.path.exists(pdf_docs_path):
        with open(pdf_docs_path, "rb") as f:
            existing_pdf_docs = pickle.load(f)
            for doc in existing_pdf_docs:
                for retrieved_doc in retrieved_docs:
                    if retrieved_doc.metadata.get("person_korean_name") == doc.metadata.get("person_korean_name"):
                        doc.metadata['RP_score'] = retrieved_doc.metadata.get("RP_score", 0)
                        matched_docs.append(doc)
    else:
        logging.warning("No existing PDF documents found in the pkl file.")
    

    if not matched_docs:
        logging.warning("No matching documents found in the pkl file.")
        return JSONResponse(content={"answer": "No matching documents found for the provided prompt."})

    logging.info(f"keywords: {jd_keywords}")
    
    # JD 관련 키워드 추출 및 점수 계산 (FP 계산)
    pdf_docs_with_fp = calculate_fp_scores(matched_docs, jd_keywords)
    
    
    pdf_docs_with_tp = calculate_tp_scores(pdf_docs_with_fp)

    logging.info(f"Number of matching documents: {len(matched_docs)}")
    for i, doc in enumerate(matched_docs, 1):
        logging.info(f"Matched Document {i}: {doc.page_content[:100]}...")

    # app.py

    # FP 점수를 기준으로 문서 정렬 (높은 순서대로)
    top_docs = sorted(pdf_docs_with_tp, key=lambda x: x.metadata.get('TP_score', 0), reverse=True)[:3]

    # 이 부분을 수정합니다.
    # final_metadata_extraction = [doc.metadata for doc in top_docs]

    try:
        logging.info("Starting LLM answer generation...")

        # LLM 호출: top_docs를 직접 전달
        answer = generate_answer_with_llm(top_docs, prompt)


    except Exception as e:
            # LLM 호출 실패에 대한 로그 기록 및 에러 메시지 반환
            logging.error(f"Error in LLM answer generation: {str(e)}")
            return JSONResponse(content={"answer": "LLM failed to generate a complete answer. Please try again later."})

    return JSONResponse(content={"answer": answer})
# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)