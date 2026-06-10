import glob
import os
import time
import traceback
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from data_processing import load_pdf, clean_text
from embeddings import OpenRouterEmbeddings
from retrieval_engine import AdvancedRetrievalEngine
from agent_engine import CloudAgentEngine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "db_new")
MEMORY_DOCS_DIR = os.path.join(BASE_DIR, "memory_documents")

# Ensure directory exists
if not os.path.exists(MEMORY_DOCS_DIR):
    os.makedirs(MEMORY_DOCS_DIR)

async def process_files_to_chroma(file_paths: List[str]):
    all_chunks = []
    processed_files_summary = []
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        try:
            raw_docs = load_pdf(file_path)
            cleaned_docs = clean_text(raw_docs)

            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200
            )
            parent_docs = parent_splitter.split_documents(cleaned_docs)

            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=30
            )

            chunks_for_file = []
            for parent_idx, parent_doc in enumerate(parent_docs):
                parent_id = f"{filename}_parent_{parent_idx}"

                filename_without_ext, _ = os.path.splitext(filename)
                child_texts = child_splitter.split_text(parent_doc.page_content)
                for child_idx, child_text in enumerate(child_texts):
                    child_doc = Document(
                        page_content=child_text,
                        metadata={
                            "parent_id": parent_id,
                            "chunk_id": f"{filename_without_ext}_parent_{parent_idx}_child_{child_idx}",
                            "parent_text": parent_doc.page_content,
                            "source": parent_doc.metadata.get("source"),
                            "page_number": parent_doc.metadata.get("page_number")
                        }
                    )
                    chunks_for_file.append(child_doc)

            all_chunks.extend(chunks_for_file)

            processed_files_summary.append(
                {
                    "filename": filename,
                    "chunks_generated": len(chunks_for_file)
                }
            )
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            continue
                
    if not all_chunks:
        return None

    try:
        embeddings = OpenRouterEmbeddings()
        
        # Batch upload to avoid openrouter rate limit blockages
        batch_size = 80
        print(f"Starting batched embedding processing for {len(all_chunks)} chunks...")
        
        initial_batch = all_chunks[:batch_size]
        vector_store = Chroma.from_documents(
            documents=initial_batch,
            embedding=embeddings,
            collection_name="pdf_embeddings",
            persist_directory=DB_DIR
        )
        
        for i in range(batch_size, len(all_chunks), batch_size):
            time.sleep(2.0)
            next_batch = all_chunks[i:i + batch_size]
            print(f"Uploading chunks {i} to {min(i + batch_size, len(all_chunks))}...")
            vector_store.add_documents(documents=next_batch)

        return {
            "status": "success",
            "message": f"Successfully vectorized and stored {len(processed_files_summary)} files.",
            "processed_files": processed_files_summary,
            "total_chunks_stored": len(all_chunks)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during embedding or storage: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """
    Optional: Automatically sync the memory folder on startup.
    """
    print("🚀 Server starting up. Checking 'memory_documents' for initial ingestion...")
    pdf_files = glob.glob(os.path.join(MEMORY_DOCS_DIR, "*.pdf"))
    if pdf_files:
        print(f"Found {len(pdf_files)} PDFs. Starting background ingestion...")
        # Note: In a production app, you might want to run this in a background task
        # to avoid blocking the server startup if there are many files.
        try:
            await ingest_static_documents()
            print("✅ Initial ingestion complete.")
        except Exception as e:
            print(f"❌ Initial ingestion failed: {e}")
    else:
        print("ℹ️ No PDFs found in 'memory_documents'. Knowledge base remains unchanged.")

@app.post("/ingest-static")
async def ingest_static_documents():
    """
    Scans the 'memory_documents' folder and ingests all PDF files into ChromaDB.
    """
    pdf_files = glob.glob(os.path.join(MEMORY_DOCS_DIR, "*.pdf"))
    if not pdf_files:
        return {"status": "info", "message": "No PDF files found in 'memory_documents' folder."}
    
    result = await process_files_to_chroma(pdf_files)
    if not result:
        raise HTTPException(status_code=400, detail="Failed to process documents or no readable text found.")
    
    return result

@app.post("/upload-pdf")
async def upload_multiple_pdf(files: List[UploadFile] = File(...)):
    """
    Accepts a list of multiple PDFs from the frontend, processes, cleans,
    chunks them, and appends them to your local ChromaDB memory bank.
    """
    temp_files = []
    
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF.")
        
        temp_file_path = os.path.join(BASE_DIR, "documents", f"temp_{file.filename}")
        try:
            contents = await file.read()
            with open(temp_file_path, "wb") as f:
                f.write(contents)
            temp_files.append(temp_file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving file {file.filename}: {str(e)}")

    if not temp_files:
        raise HTTPException(status_code=400, detail="No files provided.")

    try:
        result = await process_files_to_chroma(temp_files)
        if not result:
            raise HTTPException(status_code=400, detail="No readable text found across uploaded PDFs.")
        return result
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def advanced_agentic_query(request: QueryRequest):
    """
    Unified RAG Workflow matching the upgraded AdvancedRetrievalEngine pipeline:
    Orchestrated Retrieval -> RRF Fusion -> Relevance Filtering -> Generation & Hallucination Guard
    """
    try: 
        embeddings = OpenRouterEmbeddings()
        vector_store = Chroma(
            collection_name="pdf_embeddings",
            embedding_function=embeddings,
            persist_directory=DB_DIR
        )

        # 1. Initialize the new engine (loads key directly from environment variables)
        retrieval_engine = AdvancedRetrievalEngine(
            rewrite_model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
        )
        agent = CloudAgentEngine()

        # 2. 👉 Execute the unified pipeline. 
        # This wraps query rewriting, hybrid vector/BM25 search, RRF score sorting, and auto gap auditing.
        retrieval_result = retrieval_engine.retrieve(
            query=request.query, 
            vector_store=vector_store, 
            k=6, 
            confidence_threshold=0.10
        )

        # 3. Handle immediate early termination conditions (Empty database index or automatic low confidence)
        if retrieval_result.is_empty or retrieval_result.gap_analysis:
            # Fallback to the string generated by the audit LLM safely, ensure a default message exists
            gap_checklist = retrieval_result.gap_analysis or "The local document indexes are completely empty."
            return {
                "status": "insufficient_context",
                "actin_required": "Please supply additional context using the checklist below.",
                "action_required": "Please supply additional context using the checklist below.",
                "response": gap_checklist
            }

        # 4. Agentic Post-Retrieval Verification: Clean and verify content chunks
        valid_chunks = []
        for chunk in retrieval_result.chunks:
            # The chunk object wraps the document data inside '.content'
            grade = agent.grade_document_relevance(request.query, chunk.content)
            if "YES" in grade.upper():
                valid_chunks.append(chunk.document)

        # 5. Handle post-grading threshold failures (If chunks passed RRF confidence score but failed agent alignment checks)
        if not valid_chunks:
            # Safely generate an on-the-fly checklist if the agent drops chunks that RRF passed
            attempted_data = [{"content": c.content} for c in retrieval_result.chunks]
            gap_checklist = retrieval_engine.analyze_information_gap(request.query, attempted_data)
            
            if not gap_checklist:
                gap_checklist = "Retrieval verified potential candidates, but agent-alignment classified the context mismatch as too high."

            return {
                "status": "insufficient_context",
                "actin_required": "Please supply additional context or rephrase your question.",
                "action_required": "Please supply additional context or rephrase your question.",
                "response": gap_checklist
            }
        
        # 6. Build Context and Synthesize Answer
        context_text = "\n\n".join([doc.page_content for doc in valid_chunks])
        synthesis_prompt = f"Context:\n{context_text}\n\nQuestion: {request.query}\nAnswer the question using *only* the facts mentioned above:"

        generated_answer = agent._call_llm(agent.synthesis_model, synthesis_prompt, temperature=0.2)

        # 7. Final Security Layer: Hallucination Check
        hallucination_check = agent.grade_hallucination(context_text, generated_answer)

        if "FAILED" in hallucination_check.upper():
            return {
                "status": "agent_blocked",
                "message": "The system generated an answer, but validation checks caught an unverified fact or hallucination. Request aborted.",
                "action": "Please upload more files or rephrase your request."
            }
        
        return {
            "status": "success",
            "response": generated_answer,
            "success_utilized": [doc.metadata for doc in valid_chunks]
        }
        
    except Exception as e:
        print("\n❌ CRITICAL SYSTEM ERROR IN MAIN RUNTIME:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"RAG System error occurred: {str(e)}")