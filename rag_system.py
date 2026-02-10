import ollama
import chromadb
from pathlib import Path

# ============================================
# 1. EMBEDDING FUNCTION (FIXED AGAIN)
# ============================================
class OllamaEmbeddings:
    def __init__(self, model_name="mxbai-embed-large"):
        self.model_name = model_name
    
    def name(self):
        """Required by ChromaDB - must be a callable method"""
        return self.model_name
    
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        embeddings = []
        for text in input:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            embeddings.append(response['embedding'])
        return embeddings


# ============================================
# 2. INITIALIZE DATABASE
# ============================================
persistent_client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = OllamaEmbeddings()

collection = persistent_client.get_or_create_collection(
    name="my_rag_collection",
    embedding_function=embedding_function,
    metadata={"description": "Offline RAG with mxbai and Llama 3.2"}
)


# ============================================
# 3. UTILITY FUNCTIONS
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


def load_documents_from_directory(directory_path):
    """Load all .txt files from a directory"""
    documents = []
    metadata = []
    
    for file_path in Path(directory_path).rglob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append(content)
            metadata.append({"source": str(file_path)})
    
    return documents, metadata


def add_documents(docs, use_chunking=False):
    """Add documents to the vector database"""
    if use_chunking:
        all_chunks = []
        for doc in docs:
            chunks = chunk_text(doc)
            all_chunks.extend(chunks)
        docs = all_chunks
    
    # Get current count to generate unique IDs
    current_count = collection.count()
    
    collection.add(
        documents=docs,
        ids=[f"doc_{current_count + i}" for i in range(len(docs))]
    )
    print(f"✓ Added {len(docs)} documents to the database")


def rag_query(query, n_results=3, verbose=False):
    """Query the RAG system"""
    # Retrieve relevant documents
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    retrieved_docs = results['documents'][0]
    
    if verbose:
        print(f"\n📚 Retrieved {len(retrieved_docs)} relevant documents")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"\n--- Source {i} ---")
            print(doc[:200] + "..." if len(doc) > 200 else doc)
    
    # Build context
    context = "\n\n".join(retrieved_docs)
    
    # Create prompt
    prompt = f"""Based on the following context, answer the question. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""
    
    # Generate response
    print("\n🤔 Generating response...")
    response = ollama.generate(
        model='llama3.2',
        prompt=prompt
    )
    
    return {
        'answer': response['response'],
        'sources': retrieved_docs
    }


def show_stats():
    """Display database statistics"""
    count = collection.count()
    print(f"\n📊 Database Stats:")
    print(f"   Total documents: {count}")
    print(f"   Collection name: {collection.name}")


# ============================================
# 4. MAIN PROGRAM
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Offline RAG System with Ollama")
    print("   Models: mxbai-embed-large + Llama 3.2")
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
    print("  - 'add' - Add new documents")
    print("  - 'load' - Load documents from directory")
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
        
        elif user_input.lower() == 'load':
            dir_path = input("Enter directory path: ").strip()
            if Path(dir_path).exists():
                docs, meta = load_documents_from_directory(dir_path)
                if docs:
                    add_documents(docs, use_chunking=True)
                else:
                    print("❌ No .txt files found in directory")
            else:
                print("❌ Directory not found")
        
        elif user_input:
            # Regular query
            result = rag_query(user_input, n_results=3, verbose=False)
            print(f"\n🤖 Assistant: {result['answer']}")
            print(f"\n📎 Used {len(result['sources'])} source(s)")
        
        else:
            print("Please enter a command or question")