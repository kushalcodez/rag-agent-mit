import ollama
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import pandas as pd 

# ============================================
# 1. INITIALIZE DATABASE
# ============================================
persistent_client = chromadb.PersistentClient(path="./chroma_db")

# Use ChromaDB's built-in Ollama embedding function
embedding_function = embedding_functions.OllamaEmbeddingFunction(
    model_name="mxbai-embed-large",
    url="http://localhost:11434"
)

collection = persistent_client.get_or_create_collection(
    name="my_rag_collection",
    embedding_function=embedding_function,
    metadata={"description": "Offline RAG with mxbai and Llama 3.2"}
)


# ============================================
# 2. UTILITY FUNCTIONS
# ============================================
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks for better retrieval"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def process_csv_file(file_path, text_columns=None, combine_columns=True):
    """
    Process a CSV file and convert it to documents
    
    Args:
        file_path: Path to the CSV file
        text_columns: List of column names to use. If None, uses all columns
        combine_columns: If True, combines all columns into single documents per row.
                        If False, creates separate documents for each cell
    
    Returns:
        List of documents and their metadata
    """
    try:
        df = pd.read_csv(file_path)
        documents = []
        metadata = []
        
        # If no columns specified, use all columns
        if text_columns is None:
            text_columns = df.columns.tolist()
        else:
            # Validate columns exist
            missing_cols = set(text_columns) - set(df.columns)
            if missing_cols:
                print(f"⚠️  Warning: Columns not found: {missing_cols}")
                text_columns = [col for col in text_columns if col in df.columns]
        
        if combine_columns:
            # Create one document per row with all specified columns
            for idx, row in df.iterrows():
                doc_parts = []
                for col in text_columns:
                    value = row[col]
                    if pd.notna(value):  # Skip NaN values
                        doc_parts.append(f"{col}: {value}")
                
                if doc_parts:
                    documents.append("\n".join(doc_parts))
                    metadata.append({
                        "source": str(file_path),
                        "row_index": int(idx),
                        "type": "csv_row"
                    })
        else:
            # Create separate documents for each cell
            for idx, row in df.iterrows():
                for col in text_columns:
                    value = row[col]
                    if pd.notna(value):
                        documents.append(f"{col}: {value}")
                        metadata.append({
                            "source": str(file_path),
                            "row_index": int(idx),
                            "column": col,
                            "type": "csv_cell"
                        })
        
        return documents, metadata
        
    except Exception as e:
        print(f"❌ Error processing CSV file: {e}")
        return [], []


def load_documents_from_directory(directory_path, file_types=None):
    """
    Load documents from a directory (supports .txt and .csv files)
    
    Args:
        directory_path: Path to the directory
        file_types: List of file extensions to load (e.g., ['.txt', '.csv'])
                   If None, loads both .txt and .csv files
    """
    if file_types is None:
        file_types = ['.txt', '.csv']
    
    documents = []
    metadata = []
    
    for file_type in file_types:
        pattern = f"*{file_type}"
        
        for file_path in Path(directory_path).rglob(pattern):
            print(f"📄 Processing: {file_path.name}")
            
            if file_type == '.txt':
                # Process text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():  # Only add non-empty files
                        documents.append(content)
                        metadata.append({
                            "source": str(file_path),
                            "type": "text_file"
                        })
            
            elif file_type == '.csv':
                # Process CSV file
                csv_docs, csv_meta = process_csv_file(file_path)
                documents.extend(csv_docs)
                metadata.extend(csv_meta)
    
    return documents, metadata


def load_csv_interactive():
    """Interactive CSV loading with column selection"""
    file_path = input("Enter CSV file path: ").strip()
    
    if not Path(file_path).exists():
        print("❌ File not found")
        return [], []
    
    try:
        # Preview CSV structure
        df = pd.read_csv(file_path, nrows=5)
        print(f"\n📊 CSV Preview ({file_path}):")
        print(f"   Total columns: {len(df.columns)}")
        print(f"   Columns: {', '.join(df.columns.tolist())}\n")
        print(df.head())
        
        # Ask for column selection
        print("\n" + "=" * 50)
        col_choice = input("Enter column names to use (comma-separated) or press Enter for all: ").strip()
        
        if col_choice:
            text_columns = [col.strip() for col in col_choice.split(',')]
        else:
            text_columns = None
        
        # Ask for combination preference
        combine = input("Combine columns per row? (y/n, default=y): ").strip().lower()
        combine_columns = combine != 'n'
        
        # Process the CSV
        docs, meta = process_csv_file(file_path, text_columns, combine_columns)
        
        if docs:
            print(f"\n✓ Processed {len(docs)} documents from CSV")
            # Show sample
            print(f"\nSample document:\n{docs[0][:300]}..." if len(docs[0]) > 300 else f"\nSample document:\n{docs[0]}")
        
        return docs, meta
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return [], []


def add_documents(docs, metadata=None, use_chunking=False):
    """Add documents to the vector database"""
    if not docs:
        print("⚠️  No documents to add")
        return
    
    if use_chunking:
        all_chunks = []
        chunk_metadata = []
        for i, doc in enumerate(docs):
            chunks = chunk_text(doc)
            all_chunks.extend(chunks)
            # Replicate metadata for each chunk
            if metadata:
                for _ in chunks:
                    chunk_meta = metadata[i].copy()
                    chunk_meta['is_chunk'] = True
                    chunk_metadata.append(chunk_meta)
        docs = all_chunks
        metadata = chunk_metadata if metadata else None
    
    # Get current count to generate unique IDs
    current_count = collection.count()
    
    if metadata:
        collection.add(
            documents=docs,
            ids=[f"doc_{current_count + i}" for i in range(len(docs))],
            metadatas=metadata
        )
    else:
        collection.add(
            documents=docs,
            ids=[f"doc_{current_count + i}" for i in range(len(docs))]
        )
    
    print(f"✓ Added {len(docs)} documents to the database")


def rag_query(query, n_results=3, verbose=False):
    """Query the RAG system"""
    try:
        print("\n🔍 Searching database...")
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        retrieved_docs = results['documents'][0]
        retrieved_meta = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        if verbose:
            print(f"\n📚 Retrieved {len(retrieved_docs)} relevant documents")
            for i, (doc, meta, dist) in enumerate(zip(retrieved_docs, retrieved_meta, distances), 1):
                print(f"\n--- Source {i} (Similarity: {1-dist:.3f}) ---")
                if meta:
                    print(f"Metadata: {meta}")
                print(doc[:200] + "..." if len(doc) > 200 else doc)
        
        # Build context
        context = "\n\n".join(retrieved_docs)
        
        # Create prompt
        prompt = f"""Based on the following context, answer the question. If the context doesn't contain enough information, say so.
                     Strictly stick to the data sets provided.
Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response
        print("🤔 Generating response...")
        response = ollama.generate(
            model='llama3.2',
            prompt=prompt
        )
        
        return {
            'answer': response['response'],
            'sources': retrieved_docs,
            'metadata': retrieved_meta
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            'answer': f"Error occurred: {e}",
            'sources': [],
            'metadata': []
        }


def show_stats():
    """Display database statistics"""
    count = collection.count()
    print(f"\n📊 Database Stats:")
    print(f"   Total documents: {count}")
    print(f"   Collection name: {collection.name}")
    
    # Show metadata statistics - FIXED VERSION
    if count > 0:
        try:
            sample = collection.get(limit=min(100, count), include=['metadatas'])
            if sample.get('metadatas'):
                types = {}
                for meta in sample['metadatas']:
                    # Check if meta is a dictionary before calling .get()
                    if isinstance(meta, dict):
                        doc_type = meta.get('type', 'unknown')
                    else:
                        doc_type = 'unknown'
                    types[doc_type] = types.get(doc_type, 0) + 1
                print(f"   Document types: {types}")
        except Exception as e:
            print(f"   (Could not retrieve metadata stats: {e})")


def list_all_documents(limit=None):
    """List all documents in the database"""
    try:
        results = collection.get(limit=limit, include=['documents', 'metadatas'])
        docs = results['documents']
        ids = results['ids']
        metas = results.get('metadatas', [])
        
        print(f"\n📚 Documents in Database ({len(docs)} total):")
        print("=" * 50)
        
        for i, (doc_id, doc) in enumerate(zip(ids, docs), 1):
            print(f"\n[{i}] ID: {doc_id}")
            if metas and i <= len(metas) and metas[i-1]:
                print(f"Metadata: {metas[i-1]}")
            preview = doc[:200] + "..." if len(doc) > 200 else doc
            print(f"Content: {preview}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error listing documents: {e}")


# ============================================
# 3. MAIN PROGRAM
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Offline RAG System with Ollama")
    print("   Models: mxbai-embed-large + Llama 3.2")
    print("   Supports: TXT and CSV files")
    print("=" * 50)
    
    # Show current stats
    show_stats()
    
    # First-time setup: Add sample documents
    if collection.count() == 0:
        print("\n📝 No documents found. Adding sample documents...")
        sample_docs = [
            "Ollama is a tool for running large language models locally on your machine. It supports various models including Llama, Mistral, and others.",
            "Vector databases store high-dimensional embeddings and enable efficient similarity search. They are essential for RAG systems.",
            "RAG (Retrieval Augmented Generation) combines information retrieval with language generation. It retrieves relevant context before generating responses.",
            "Llama 3.2 is Meta's latest open-source language model, available in various sizes for different use cases.",
            "mxbai-embed-large is an embedding model optimized for retrieval tasks, producing 1024-dimensional vectors.",
            "ChromaDB is an open-source vector database designed for AI applications. It stores embeddings and metadata together."
        ]
        add_documents(sample_docs)
    
    # Interactive query loop
    print("\n" + "=" * 50)
    print("💬 Interactive Mode")
    print("=" * 50)
    print("\nCommands:")
    print("  - Type your question to query the system")
    print("  - 'add' - Add new documents manually")
    print("  - 'csv' - Load and process a CSV file")
    print("  - 'load' - Load documents from directory (.txt and .csv)")
    print("  - 'list' - List all documents")
    print("  - 'stats' - Show database statistics")
    print("  - 'quit' - Exit the program")
    
    while True:
        print("\n" + "-" * 50)
        user_input = input("👤 You: ").strip()
        
        if user_input.lower() == 'quit':
            print("\n👋 Goodbye!")
            break
        
        elif user_input.lower() == 'stats':
            show_stats()
        
        elif user_input.lower() == 'list':
            limit = input("How many documents to show? (press Enter for all): ").strip()
            limit = int(limit) if limit else None
            list_all_documents(limit)
        
        elif user_input.lower() == 'add':
            print("\nEnter documents (one per line, empty line to finish):")
            new_docs = []
            while True:
                doc = input()
                if not doc:
                    break
                new_docs.append(doc)
            if new_docs:
                add_documents(new_docs)
        
        elif user_input.lower() == 'csv':
            docs, meta = load_csv_interactive()
            if docs:
                use_chunk = input("Apply text chunking? (y/n, default=n): ").strip().lower()
                add_documents(docs, meta, use_chunking=(use_chunk == 'y'))
        
        elif user_input.lower() == 'load':
            dir_path = input("Enter directory path: ").strip()
            if Path(dir_path).exists():
                file_types = input("File types to load (comma-separated, e.g., '.txt,.csv' or Enter for both): ").strip()
                if file_types:
                    file_types = [ft.strip() for ft in file_types.split(',')]
                else:
                    file_types = None
                
                docs, meta = load_documents_from_directory(dir_path, file_types)
                if docs:
                    use_chunk = input("Apply text chunking? (y/n, default=y): ").strip().lower()
                    add_documents(docs, meta, use_chunking=(use_chunk != 'n'))
                else:
                    print("❌ No files found in directory")
            else:
                print("❌ Directory not found")
        
        elif user_input:
            # Regular query
            verbose = input("Show detailed sources? (y/n, default=n): ").strip().lower() == 'y'
            result = rag_query(user_input, n_results=3, verbose=verbose)
            print(f"\n🤖 Assistant: {result['answer']}")
            print(f"\n📎 Used {len(result['sources'])} source(s)")
        
        else:
            print("Please enter a command or question")