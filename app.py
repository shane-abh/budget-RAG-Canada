import os
import time
import json
import asyncio
import uuid
from typing import Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

# ==================== FastAPI Setup ====================
app = FastAPI(title="RAG Chat API", version="1.0.0")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Request/Response Models ====================
class ChatMessage(BaseModel):
    session_id: Optional[str] = None
    message: str
    use_query_expansion: bool = True

class SessionResponse(BaseModel):
    session_id: str

class DocumentUploadResponse(BaseModel):
    message: str
    document_name: str
    num_pages: int
    num_chunks: int
    processing_time: float

# ==================== Global Variables ====================
vector_db = None
ensemble_retriever = None
vector_retriever = None
bm25_retriever = None
llm = None
emb = None
pc = None
index_name = "rag-documents"

# Session storage
conversation_sessions: Dict[str, List[Dict]] = {}

# ==================== Helper Functions (from your original code) ====================
INDEXED_DOCS_FILE = "indexed_documents.json"

def load_indexed_documents():
    """Load the list of already indexed documents"""
    if os.path.exists(INDEXED_DOCS_FILE):
        with open(INDEXED_DOCS_FILE, 'r') as f:
            return json.load(f)
    return {"documents": [], "index_name": None}

def save_indexed_documents(doc_list, index_name):
    """Save the list of indexed documents"""
    with open(INDEXED_DOCS_FILE, 'w') as f:
        json.dump({"documents": doc_list, "index_name": index_name}, f, indent=2)
    print(f"✓ Saved indexed documents list to {INDEXED_DOCS_FILE}")

def rewrite_query(original_query):
    """Rewrite the query to be clearer and more specific"""
    rewrite_prompt = f"""Rewrite the following query to be more specific and clear for document retrieval. 
Keep the core meaning but make it more precise. Return only the rewritten query, nothing else.

Original query: {original_query}

Rewritten query:"""
    
    response = llm.invoke(rewrite_prompt)
    return response.content.strip()

def expand_query(query):
    """Generate synonyms and related terms for query expansion"""
    expand_prompt = f"""Given the following query, generate 5-8 synonyms, related terms, or alternative phrasings 
that would help find relevant documents. Focus on key concepts and terms.
Return only a comma-separated list of terms, nothing else.

Query: {query}

Related terms:"""
    
    response = llm.invoke(expand_prompt)
    terms = [term.strip() for term in response.content.strip().split(",")]
    return terms

def generate_query_variations(original_query, rewritten_query, expanded_terms):
    """Generate multiple query variations for retrieval"""
    variations = [original_query, rewritten_query]
    
    # Create variations by combining rewritten query with expanded terms
    key_terms = expanded_terms[:5]  # Use top 5 expanded terms
    
    # Add variations that incorporate expanded terms
    for term in key_terms:
        if term.lower() not in rewritten_query.lower():
            variations.append(f"{rewritten_query} {term}")
    
    # Add some standalone expanded term combinations
    if len(key_terms) >= 2:
        variations.append(f"{key_terms[0]} {key_terms[1]}")
    
    return list(set(variations))  # Remove duplicates

def retrieve_with_expansion(question, max_docs_per_query=10):
    """Retrieve documents using query expansion and rewriting"""
    print(f"\n[Query Processing] Original: {question}")
    
    # Use the standard ensemble retriever
    retriever_to_use = ensemble_retriever
    
    # Step 1: Rewrite query
    print("[Query Processing] Step 1: Rewriting query...")
    rewritten = rewrite_query(question)
    print(f"[Query Processing] ✓ Rewritten: {rewritten}")
    
    # Step 2: Expand query
    print("[Query Processing] Step 2: Expanding query with synonyms...")
    expanded_terms = expand_query(rewritten)
    print(f"[Query Processing] ✓ Expanded terms ({len(expanded_terms)} total): {expanded_terms[:5]}...")
    
    # Step 3: Generate variations
    print("[Query Processing] Step 3: Generating query variations...")
    variations = generate_query_variations(question, rewritten, expanded_terms)
    print(f"[Query Processing] ✓ Generated {len(variations)} query variations")
    print(f"  - Variations: {variations[:3]}..." if len(variations) > 3 else f"  - Variations: {variations}")
    
    # Step 4: Retrieve from all variations
    print(f"\n[Retrieval] Step 4: Retrieving documents from {min(5, len(variations))} query variations...")
    all_docs = []
    seen_content = set()  # For deduplication
    
    for i, variation in enumerate(variations[:5]):  # Limit to 5 variations to avoid too many API calls
        try:
            print(f"  - Variation {i+1}/{min(5, len(variations))}: '{variation[:50]}...'")
            docs = retriever_to_use.invoke(variation)
            print(f"    → Retrieved {len(docs)} documents")
            for doc in docs:
                # Deduplicate by content
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)
            print(f"    → After deduplication: {len(all_docs)} unique documents")
        except Exception as e:
            print(f"    ⚠ Warning: Error retrieving for variation '{variation}': {e}")
            continue
    
    # Step 5: Limit total documents (keep most relevant)
    print(f"\n[Retrieval] Step 5: Limiting to top {max_docs_per_query * 2} documents...")
    final_docs = all_docs[:max_docs_per_query * 2]  # Get more, then we'll refine
    
    print(f"[Retrieval] ✓ Final result: {len(final_docs)} unique documents from {len(variations)} query variations")
    print(f"  - Total context length: {sum(len(doc.page_content) for doc in final_docs):,} characters")
    
    return final_docs, rewritten, expanded_terms, variations

def merge_semantic_chunks(pre_chunks, embeddings_model, similarity_threshold=0.7):
    """Merge pre-chunks based on semantic similarity"""
    overall_start = time.time()
    
    print("\n" + "=" * 80)
    print("[STEP 5] Semantic Chunking - Merging Related Chunks")
    print("=" * 80)
    print(f"Input: {len(pre_chunks)} pre-chunks to process")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Max chunk size: 2800 characters")
    
    if len(pre_chunks) < 2:
        print("⚠ Warning: Less than 2 pre-chunks, returning as-is")
        return pre_chunks
    
    # Get embeddings for all pre-chunks
    print(f"\n[5.1] Generating embeddings for {len(pre_chunks)} pre-chunks...")
    texts = [chunk.page_content for chunk in pre_chunks]
    total_text_length = sum(len(t) for t in texts)
    print(f"  - Total text to embed: {total_text_length:,} characters")
    print("  - Calling embedding model...")
    embedding_start = time.time()
    embeddings = embeddings_model.embed_documents(texts)
    embedding_time = time.time() - embedding_start
    print(f"✓ Generated {len(embeddings)} embeddings")
    print(f"  - Embedding dimension: {len(embeddings[0])}")
    print(f"  ⏱ Embedding generation time: {embedding_time:.2f} seconds ({embedding_time/60:.2f} minutes)")
    print(f"  ⚡ Embedding speed: {len(pre_chunks)/embedding_time:.2f} chunks/second")
    
    merged_chunks = []
    current_chunk = [pre_chunks[0]]
    current_text = pre_chunks[0].page_content
    merge_count = 0
    split_count = 0
    
    print(f"\n[5.2] Processing similarities and merging chunks...")
    print(f"  - Processing {len(pre_chunks) - 1} similarity comparisons...")
    similarity_start = time.time()
    
    for i in range(1, len(pre_chunks)):
        # Calculate cosine similarity
        emb1 = np.array(embeddings[i-1])
        emb2 = np.array(embeddings[i])
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        current_size = len(current_text)
        next_size = len(pre_chunks[i].page_content)
        would_exceed = current_size + next_size >= 2800
        
        # Log every 10th comparison or when interesting things happen
        if i % 10 == 0 or similarity < similarity_threshold or would_exceed:
            print(f"  - Chunk {i-1} -> {i}: similarity={similarity:.3f}, "
                  f"current_size={current_size}, next_size={next_size}, "
                  f"would_exceed={would_exceed}")
        
        # Merge if similar and within size limit
        if similarity >= similarity_threshold and not would_exceed:
            current_chunk.append(pre_chunks[i])
            current_text += "\n\n" + pre_chunks[i].page_content
            merge_count += 1
        else:
            # Save current chunk and start new one
            reason = "size_limit" if would_exceed else "low_similarity"
            if i % 10 == 0 or similarity < similarity_threshold:
                print(f"    → Creating new chunk (reason: {reason})")
            merged_chunks.append(Document(
                page_content=current_text,
                metadata={
                    **current_chunk[0].metadata,
                    "chunk_type": "merged" if len(current_chunk) > 1 else "single",
                    "num_source_chunks": len(current_chunk),
                    "chunk_size": len(current_text),
                    "merge_reason": reason,
                    "page_start": min(chunk.metadata.get("page", 0) for chunk in current_chunk),
                    "page_end": max(chunk.metadata.get("page", 0) for chunk in current_chunk),
                    "source_file": current_chunk[0].metadata.get("source_file", "unknown"),
                    "source_type": "pdf",
                    "created_at": time.time(),
                    "word_count": len(current_text.split()),
                    "chunk_index": len(merged_chunks),
                }
            ))
            current_chunk = [pre_chunks[i]]
            current_text = pre_chunks[i].page_content
            split_count += 1
    
    similarity_time = time.time() - similarity_start
    print(f"  ⏱ Similarity calculation & merging time: {similarity_time:.2f} seconds")
    print(f"  ⚡ Processing speed: {(len(pre_chunks)-1)/similarity_time:.2f} comparisons/second")
    
    # Add last chunk
    if current_chunk:
        merged_chunks.append(Document(
            page_content=current_text,
            metadata=current_chunk[0].metadata
        ))
    
    overall_time = time.time() - overall_start
    print(f"\n✓ Semantic merging complete!")
    print(f"  - Input pre-chunks: {len(pre_chunks)}")
    print(f"  - Output merged chunks: {len(merged_chunks)}")
    print(f"  - Merges performed: {merge_count}")
    print(f"  - Splits performed: {split_count}")
    avg_chunk_size = sum(len(c.page_content) for c in merged_chunks) // len(merged_chunks) if merged_chunks else 0
    largest_chunk = max(len(c.page_content) for c in merged_chunks) if merged_chunks else 0
    smallest_chunk = min(len(c.page_content) for c in merged_chunks) if merged_chunks else 0
    print(f"  - Average merged chunk size: {avg_chunk_size:,} characters")
    print(f"  - Largest merged chunk: {largest_chunk:,} characters")
    print(f"  - Smallest merged chunk: {smallest_chunk:,} characters")
    print(f"\n⏱ TOTAL SEMANTIC CHUNKING TIME: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    
    return merged_chunks

# ==================== Session Management ====================
def create_session() -> str:
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    conversation_sessions[session_id] = []
    return session_id

def add_to_session(session_id: str, role: str, content: str, metadata: dict = None):
    """Add a message to session history"""
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = []
    
    conversation_sessions[session_id].append({
        "role": role,
        "content": content,
        "timestamp": time.time(),
        "metadata": metadata or {}
    })

def get_session_history(session_id: str, last_n: int = 3) -> str:
    """Get formatted conversation history for context"""
    if session_id not in conversation_sessions:
        return ""
    
    history = conversation_sessions[session_id][-last_n:]
    if not history:
        return ""
    
    formatted = "Previous conversation:\n"
    for msg in history:
        formatted += f"{msg['role'].upper()}: {msg['content']}\n"
    return formatted + "\n"

# ==================== Startup Event ====================
@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global vector_db, ensemble_retriever, vector_retriever, bm25_retriever, llm, emb, pc
    
    print("\n" + "=" * 80)
    print("🚀 INITIALIZING RAG SYSTEM")
    print("=" * 80)
    
    # Load environment variables
    load_dotenv()
    print("✓ Environment variables loaded from .env file")
    
    # Initialize embedding model
    print("\n[STEP 1] Initializing Embedding Model")
    emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print(f"✓ Embedding model initialized: models/embedding-001")
    
    # Initialize Pinecone
    print("\n[STEP 2] Initializing Pinecone")
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    print(f"✓ Pinecone client initialized")
    print(f"  - Index name: {index_name}")
    
    # Connect to existing index
    existing_indexes = [index.name for index in pc.list_indexes()]
    print(f"  - Existing indexes: {existing_indexes}")
    
    if index_name in existing_indexes:
        print(f"\n[STEP 3] Connecting to Pinecone Index")
        vector_db = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=emb
        )
        print("✓ Connected to existing Pinecone index")
        
        # Setup retrievers
        print("\n[STEP 4] Setting up Retrievers")
        vector_retriever = vector_db.as_retriever(search_kwargs={"k": 10})
        ensemble_retriever = vector_retriever  # Using vector retriever for now
        print("✓ Vector retriever created (k=10)")
        print("⚠ Note: BM25 retriever will be initialized per-session or when documents are added")
    else:
        print("⚠ WARNING: No existing Pinecone index found!")
        print("  Please upload documents to create the index.")
    
    # Initialize LLM
    print("\n[STEP 5] Initializing LLM")
    llm = init_chat_model("google_genai:gemini-2.5-flash-lite")
    print("✓ LLM initialized: google_genai:gemini-2.5-flash-lite")
    
    print("\n" + "=" * 80)
    print("✅ RAG SYSTEM READY")
    print("=" * 80)

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "RAG Chat API is running",
        "version": "1.0.0"
    }

@app.post("/session/new", response_model=SessionResponse)
async def new_session():
    """Create a new chat session"""
    session_id = create_session()
    return SessionResponse(session_id=session_id)

@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in conversation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "messages": conversation_sessions[session_id],
        "total_messages": len(conversation_sessions[session_id])
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
        return {"message": "Session deleted successfully"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.post("/chat/stream")
async def chat_stream_post(chat_message: ChatMessage):
    """
    Stream chat response with query optimization display (POST method).
    Shows original query, rewritten query, expanded terms, and streams the answer.
    """
    return await _chat_stream_handler(
        session_id=chat_message.session_id,
        message=chat_message.message,
        use_query_expansion=chat_message.use_query_expansion
    )

@app.get("/chat/stream")
async def chat_stream_get(
    session_id: Optional[str] = None,
    message: str = "",
    use_query_expansion: bool = True
):
    """
    Stream chat response with query optimization display (GET method).
    Accepts query parameters instead of request body.
    """
    if not message:
        raise HTTPException(status_code=400, detail="Message parameter is required")
    
    return await _chat_stream_handler(
        session_id=session_id,
        message=message,
        use_query_expansion=use_query_expansion
    )

async def _chat_stream_handler(
    session_id: Optional[str],
    message: str,
    use_query_expansion: bool
):
    """
    Internal handler for chat streaming logic.
    Used by both GET and POST endpoints.
    """
    if ensemble_retriever is None or llm is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized. Please upload documents first.")
    
    # Get or create session
    session_id = session_id or create_session()
    question = message
    use_query_expansion = use_query_expansion
    
    async def event_generator():
        try:
            # Send session info
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
            await asyncio.sleep(0.05)
            
            # Send original query
            yield f"data: {json.dumps({'type': 'original_query', 'content': question})}\n\n"
            await asyncio.sleep(0.1)
            
            if use_query_expansion:
                # Query optimization process
                yield f"data: {json.dumps({'type': 'status', 'message': 'Optimizing query...'})}\n\n"
                
                # Rewrite query
                yield f"data: {json.dumps({'type': 'status', 'message': 'Step 1: Rewriting query...'})}\n\n"
                rewritten = rewrite_query(question)
                yield f"data: {json.dumps({'type': 'rewritten_query', 'content': rewritten})}\n\n"
                await asyncio.sleep(0.1)
                
                # Expand query
                yield f"data: {json.dumps({'type': 'status', 'message': 'Step 2: Expanding query terms...'})}\n\n"
                expanded_terms = expand_query(rewritten)
                yield f"data: {json.dumps({'type': 'expanded_terms', 'terms': expanded_terms})}\n\n"
                await asyncio.sleep(0.1)
                
                # Generate variations
                yield f"data: {json.dumps({'type': 'status', 'message': 'Step 3: Generating query variations...'})}\n\n"
                variations = generate_query_variations(question, rewritten, expanded_terms)
                yield f"data: {json.dumps({'type': 'query_variations', 'variations': variations[:3]})}\n\n"
                await asyncio.sleep(0.1)
                
                # Retrieve documents with expansion
                yield f"data: {json.dumps({'type': 'status', 'message': 'Step 4: Retrieving relevant documents...'})}\n\n"
                docs, rewritten_used, expanded_used, variations_used = retrieve_with_expansion(question)
                
                metadata = {
                    "rewritten": rewritten_used,
                    "expanded_terms": expanded_used,
                    "variations": variations_used
                }
            else:
                # Simple retrieval without expansion
                yield f"data: {json.dumps({'type': 'status', 'message': 'Retrieving documents...'})}\n\n"
                docs = ensemble_retriever.invoke(question)
                metadata = {}
            
            # Send source information
            sources = []
            for doc in docs[:5]:
                sources.append({
                    "page": doc.metadata.get("page", "unknown"),
                    "source_file": doc.metadata.get("source_file", "unknown"),
                    "preview": doc.page_content[:200] + "..."
                })
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'total_docs': len(docs)})}\n\n"
            await asyncio.sleep(0.1)
            
            # Get conversation history
            history_context = get_session_history(session_id)
            
            # Prepare prompt
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""{history_context}Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know. Give detailed explanation for the answer.

Context:
{context}

Question: {question}

Answer:"""
            
            # Generate answer
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating answer...'})}\n\n"
            yield f"data: {json.dumps({'type': 'answer_start'})}\n\n"
            
            # Get LLM response
            response = llm.invoke(prompt)
            answer_text = response.content
            
            # Stream the answer word by word for better UX
            words = answer_text.split()
            for i, word in enumerate(words):
                chunk = word + " "
                yield f"data: {json.dumps({'type': 'answer_chunk', 'content': chunk})}\n\n"
                # Add small delay for streaming effect
                if i % 5 == 0:  # Only delay every 5 words to be faster
                    await asyncio.sleep(0.02)
            
            yield f"data: {json.dumps({'type': 'answer_complete'})}\n\n"
            
            # Save to session history
            add_to_session(session_id, "user", question, metadata)
            add_to_session(session_id, "assistant", answer_text)
            
            # Send completion
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            print(f"Error in chat_stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a new PDF document.
    Uses the same semantic chunking process as the original script.
    """
    global vector_db, ensemble_retriever
    
    if emb is None or pc is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"\n{'='*80}")
        print(f"Processing uploaded document: {file.filename}")
        print(f"{'='*80}")
        
        # Load PDF
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        
        # Add source file metadata
        for page in pages:
            page.metadata["source_file"] = file.filename
        
        print(f"✓ Loaded {len(pages)} pages from {file.filename}")
        
        # Pre-split
        pre_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        pre_chunks = pre_splitter.split_documents(pages)
        print(f"✓ Pre-split into {len(pre_chunks)} chunks")
        
        # Semantic chunking
        chunks = merge_semantic_chunks(pre_chunks, emb, similarity_threshold=0.7)
        print(f"✓ Semantic chunking complete: {len(chunks)} final chunks")
        
        # Check if index exists, create if not
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"Creating new Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(5)  # Wait for index to be ready
            
            vector_db = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=emb,
                index_name=index_name
            )
            print(f"✓ Created index and added documents")
        else:
            if vector_db is None:
                vector_db = PineconeVectorStore.from_existing_index(
                    index_name=index_name,
                    embedding=emb
                )
            
            vector_db.add_documents(chunks)
            print(f"✓ Added documents to existing index")
        
        # Update retrievers
        vector_retriever = vector_db.as_retriever(search_kwargs={"k": 10})
        ensemble_retriever = vector_retriever
        
        # Update indexed documents tracker
        indexed_info = load_indexed_documents()
        indexed_documents = indexed_info.get("documents", [])
        if file.filename not in indexed_documents:
            indexed_documents.append(file.filename)
            save_indexed_documents(indexed_documents, index_name)
        
        # Clean up temp file
        os.remove(temp_path)
        
        processing_time = time.time() - start_time
        
        print(f"✓ Document indexed successfully in {processing_time:.2f} seconds")
        print(f"{'='*80}\n")
        
        return DocumentUploadResponse(
            message="Document uploaded and indexed successfully",
            document_name=file.filename,
            num_pages=len(pages),
            num_chunks=len(chunks),
            processing_time=processing_time
        )
        
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.get("/documents")
async def list_documents():
    """Get list of all indexed documents"""
    indexed_info = load_indexed_documents()
    return {
        "index_name": indexed_info.get("index_name"),
        "documents": indexed_info.get("documents", []),
        "total": len(indexed_info.get("documents", []))
    }

# ==================== Run Server ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

