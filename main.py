import os
import time
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()
print("✓ Environment variables loaded from .env file")

# Track indexed documents
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



# 3. Embedding model
print("\n" + "=" * 80)
print("[STEP 1] Initializing Embedding Model")
print("=" * 80)
emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print(f"✓ Embedding model initialized: models/embedding-001")

# 4. Initialize Pinecone
print("\n" + "=" * 80)
print("[STEP 1.5] Initializing Pinecone")
print("=" * 80)
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "rag-documents"  # You can change this to your preferred index name
print(f"✓ Pinecone client initialized")
print(f"  - Index name: {index_name}")

# Load indexed documents tracker
print("\n" + "=" * 80)
print("[STEP 1.75] Checking Indexed Documents")
print("=" * 80)
indexed_info = load_indexed_documents()
indexed_documents = set(indexed_info.get("documents", []))
stored_index_name = indexed_info.get("index_name")

print(f"Indexed documents file: {INDEXED_DOCS_FILE}")
if indexed_documents:
    print(f"✓ Found {len(indexed_documents)} previously indexed documents:")
    for doc in indexed_documents:
        print(f"  - {doc}")
    print(f"  - Index: {stored_index_name}")
else:
    print("  - No previously indexed documents found")

# List of documents to load (process one at a time to avoid quota limits)
documents_to_load = [
    "document2.pdf",
    # "2025-Summary Report and Consolidated Financial Statement.pdf"  # Uncomment to process this next
]

# Check which documents need to be processed
new_documents = [doc for doc in documents_to_load if doc not in indexed_documents]
already_indexed = [doc for doc in documents_to_load if doc in indexed_documents]

print("\n" + "=" * 80)
print("[STEP 2] Document Processing Analysis")
print("=" * 80)
print(f"Total documents requested: {len(documents_to_load)}")
print(f"Already indexed: {len(already_indexed)}")
if already_indexed:
    for doc in already_indexed:
        print(f"  ✓ {doc}")
print(f"New documents to process: {len(new_documents)}")
if new_documents:
    for doc in new_documents:
        print(f"  → {doc}")

# Only process new documents
chunks = []
if new_documents:
    print("\n" + "=" * 80)
    print("[STEP 3] Loading New Documents")
    print("=" * 80)
    
    all_pages = []
    for doc_path in new_documents:  # Only process NEW documents
        print(f"\nLoading: {doc_path}")
        loader = PyPDFLoader(doc_path)
        print(f"  ✓ PDF loader created")
        pages = loader.load()
        
        # Add source file metadata to each page
        for page in pages:
            page.metadata["source_file"] = doc_path
        
        all_pages.extend(pages)
        print(f"  ✓ Loaded {len(pages)} pages")
        print(f"    - Total characters: {sum(len(p.page_content) for p in pages):,}")
        if pages:
            print(f"    - Average page length: {sum(len(p.page_content) for p in pages) // len(pages):,} characters")

    pages = all_pages
    print(f"\n{'='*80}")
    print(f"✓ Total new documents loaded: {len(new_documents)}")
    print(f"✓ Total pages loaded: {len(pages)}")
    print(f"  - Total characters: {sum(len(p.page_content) for p in pages):,}")
    print(f"  - Average page length: {sum(len(p.page_content) for p in pages) // len(pages):,} characters")

    text = "\n".join([p.page_content for p in pages])
    print(f"  - Combined text length: {len(text):,} characters")

    # 2. Chunk
    print("\n" + "=" * 80)
    print("[STEP 4] Pre-Splitting Documents")
    print("=" * 80)
    print("Initializing RecursiveCharacterTextSplitter...")
    print("  - chunk_size: 1000")
    print("  - chunk_overlap: 100")
    print("  - separators: ['\\n\\n', '\\n', '. ', ' ', '']")
    pre_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Smaller initial chunks
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    print("✓ Splitter initialized")
    print("Starting pre-splitting...")
    start_pre_split = time.time()
    pre_chunks = pre_splitter.split_documents(pages)
    pre_split_time = time.time() - start_pre_split
    print(f"✓ Pre-splitting complete: {len(pre_chunks)} pre-chunks created")
    print(f"  ⏱ Pre-splitting time: {pre_split_time:.2f} seconds ({pre_split_time/60:.2f} minutes)")
    print(f"  - Average pre-chunk size: {sum(len(c.page_content) for c in pre_chunks) // len(pre_chunks):,} characters")
    print(f"  - Largest pre-chunk: {max(len(c.page_content) for c in pre_chunks):,} characters")
    print(f"  - Smallest pre-chunk: {min(len(c.page_content) for c in pre_chunks):,} characters")

    # Then apply semantic chunking to merge related chunks
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
        
        import numpy as np
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
                    metadata=
                    {
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
        print(f"  - Embedding generation: {embedding_time:.2f}s ({embedding_time/overall_time*100:.1f}%)")
        print(f"  - Similarity & merging: {similarity_time:.2f}s ({similarity_time/overall_time*100:.1f}%)")
        
        return merged_chunks

    print("\nStarting semantic chunking process...")
    chunking_start = time.time()
    chunks = merge_semantic_chunks(pre_chunks, emb, similarity_threshold=0.7)
    chunking_total_time = time.time() - chunking_start
    print(f"\n✓ Final chunks ready: {len(chunks)} chunks")
    print(f"\n{'='*80}")
    print(f"📊 CHUNKING PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"Pre-splitting time:     {pre_split_time:>8.2f} seconds ({pre_split_time/60:>6.2f} minutes)")
    print(f"Semantic chunking time: {chunking_total_time:>8.2f} seconds ({chunking_total_time/60:>6.2f} minutes)")
    print(f"{'─'*80}")
    print(f"Total chunking time:     {pre_split_time + chunking_total_time:>8.2f} seconds ({(pre_split_time + chunking_total_time)/60:>6.2f} minutes)")
    print(f"{'='*80}")
else:
    print("\n" + "=" * 80)
    print("[STEP 3-5] Skipping Document Processing")
    print("=" * 80)
    print("✓ All requested documents are already indexed")
    print("✓ Skipping: Loading, Pre-splitting, and Semantic chunking")
    print("✓ This saves significant time and API costs!")
    print("=" * 80)


# Check if Pinecone index exists, create if not
print("\n" + "=" * 80)
print("[STEP 6] Vector Database Setup (Pinecone)")
print("=" * 80)

# Check if index exists
existing_indexes = [index.name for index in pc.list_indexes()]
print(f"  - Existing indexes: {existing_indexes}")

if new_documents:
    # We have new documents to add
    if index_name not in existing_indexes:
        # Create new index
        print(f"  Creating new Pinecone index: {index_name}")
        print(f"  - Dimension: 768 (for models/embedding-001)")
        print(f"  - Metric: cosine")
        print(f"  - Cloud: AWS, Region: us-east-1")
        
        pc.create_index(
            name=index_name,
            dimension=768,  # Google's embedding-001 model produces 768-dimensional embeddings
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Choose region closest to you
            )
        )
        print(f"✓ Index created: {index_name}")
        
        # Wait for index to be ready
        print("  - Waiting for index to be ready...")
        time.sleep(5)  # Give it a moment to initialize
        
        # Create vector store and add documents
        print(f"  - Adding {len(chunks)} chunks to index...")
        vector_db = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=emb,
            index_name=index_name
        )
        print(f"✓ Documents added to Pinecone index")
    else:
        # Load existing index
        print(f"✓ Existing index found: {index_name}")
        print("  - Connecting to existing Pinecone index...")
        vector_db = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=emb
        )
        print("✓ Connected to existing Pinecone index")
        
        # Add new documents to existing index
        print(f"\n  - Adding {len(chunks)} new chunks to existing index...")
        add_start = time.time()
        vector_db.add_documents(chunks)
        add_time = time.time() - add_start
        print(f"✓ Documents added to Pinecone index")
        print(f"  ⏱ Upload time: {add_time:.2f} seconds ({add_time/60:.2f} minutes)")
    
    # Update indexed documents list
    print(f"\n  - Updating indexed documents tracker...")
    all_indexed_docs = list(indexed_documents) + new_documents
    save_indexed_documents(all_indexed_docs, index_name)
else:
    # No new documents, just connect to existing index
    if index_name in existing_indexes:
        print(f"✓ Existing index found: {index_name}")
        print("  - Connecting to existing Pinecone index...")
        vector_db = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=emb
        )
        print("✓ Connected to existing Pinecone index")
        print("✓ No new documents to add")
    else:
        print("⚠ WARNING: Index does not exist and no documents to add!")
        print("  Please add documents to create the index first.")
        exit(1)

# 5. Create hybrid retriever (semantic + keyword)
print("\n" + "=" * 80)
print("[STEP 7] Retriever Setup")
print("=" * 80)

# Vector retriever (semantic/dense search)
print("Creating vector retriever (semantic/dense search)...")
vector_retriever = vector_db.as_retriever(search_kwargs={"k": 10})
print("✓ Vector retriever created (k=10)")

# BM25 retriever (keyword/sparse search)
# Note: BM25 is in-memory and needs documents. If no new docs, we'll skip BM25 for now.
# In a production system, you'd want to persist and reload all chunks for BM25.
if new_documents:
    print("Creating BM25 retriever (keyword/sparse search)...")
    print(f"  - Indexing {len(chunks)} new chunks...")
    bm25_retriever = BM25Retriever.from_documents(chunks, k=10)
    print("✓ BM25 retriever created (k=10)")
    
    # Ensemble retriever combines both
    print("Creating ensemble retriever...")
    print("  - Combining vector retriever (weight: 0.5)")
    print("  - Combining BM25 retriever (weight: 0.5)")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]  # Equal weight to both retrievers (adjust as needed)
    )
    print("✓ Ensemble retriever created")
else:
    print("⚠ Skipping BM25 retriever (no new documents processed)")
    print("  - BM25 requires in-memory chunks which weren't loaded")
    print("  - Using vector retriever only for this run")
    print("  - For full functionality, consider persisting chunks for BM25")
    
    # Use vector retriever as the ensemble retriever
    bm25_retriever = None
    ensemble_retriever = vector_retriever  # Just use vector retriever
    print("✓ Using vector retriever as primary retriever")


# 6. LLM
print("\n" + "=" * 80)
print("[STEP 8] LLM Initialization")
print("=" * 80)
print("Initializing chat model: google_genai:gemini-2.5-flash-lite")
llm = init_chat_model("google_genai:gemini-2.5-flash-lite")
print("✓ LLM initialized")

# 7. Query Expansion and Rewriting Functions
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
    # The ensemble retriever already ranks them, so we take top results
    print(f"\n[Retrieval] Step 5: Limiting to top {max_docs_per_query * 2} documents...")
    final_docs = all_docs[:max_docs_per_query * 2]  # Get more, then we'll refine
    
    print(f"[Retrieval] ✓ Final result: {len(final_docs)} unique documents from {len(variations)} query variations")
    print(f"  - Total context length: {sum(len(doc.page_content) for doc in final_docs):,} characters")
    
    return final_docs

# 9. RAG Chain with Query Expansion
def ask_question(question, use_query_expansion=True):
    """
    Ask a question using RAG with optional query expansion.
    
    Args:
        question: The user's question
        use_query_expansion: Whether to use query expansion and rewriting (default: True)
    """
    print("\n" + "=" * 80)
    print("[RAG Pipeline] Starting Question Answering")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Query expansion: {'Enabled' if use_query_expansion else 'Disabled'}")
    
    if use_query_expansion:
        # Retrieve with query expansion
        docs = retrieve_with_expansion(question)
    else:
        # Original simple retrieval
        print(f"\n[Retrieval] Simple retrieval (no expansion)...")
        docs = ensemble_retriever.invoke(question)
        print(f"[Retrieval] ✓ Retrieved {len(docs)} documents")
        print(f"  - Total context length: {sum(len(doc.page_content) for doc in docs):,} characters")
    
    # Combine context
    print(f"\n[RAG Pipeline] Combining context from {len(docs)} documents...")
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"  - Combined context length: {len(context):,} characters")
    
    # Create prompt
    print(f"[RAG Pipeline] Generating prompt for LLM...")
    prompt = f"""Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know. Give detailed explanation for the answer.


Context:
{context}

Question: {question}

Answer:"""
    
    # Get response from LLM
    print(f"[RAG Pipeline] Sending prompt to LLM (prompt length: {len(prompt):,} characters)...")
    response = llm.invoke(prompt)
    print(f"[RAG Pipeline] ✓ Received response from LLM")
    print(f"  - Response length: {len(response.content):,} characters")
    
    return response.content

# 9. Ask questions
print("\n" + "=" * 80)
print("RAG PIPELINE with Query Expansion")
print("=" * 80)

answer = ask_question("what is Canada's response to U.S. tariffs on Canadian goods?", use_query_expansion=True)
print("\n" + "=" * 80)
print("RAG ANSWER:")
print("=" * 80)
print(answer)