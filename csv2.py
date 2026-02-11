import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

class CSVRAGSystem:
    def __init__(self, csv_path: str, model: str = "llama3.2:3b"):
        self.csv_path = csv_path
        self.model = model
        self.persist_directory = "./chroma_csv_db"
        
        print("\n" + "="*60)
        print(f"🚀 CSV RAG System")
        print("="*60 + "\n")
        
        # Load CSV
        print(f"📊 Loading CSV: {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
        print(f"   Columns: {', '.join(self.df.columns.tolist())}\n")
        
        # Show preview
        print("📋 First 3 rows:")
        print(self.df.head(3).to_string(index=False))
        print()
        
        # Initialize embeddings and LLM
        print("🤖 Loading AI models...")
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.llm = OllamaLLM(model=model, temperature=0)
        print("✅ Models ready\n")
        
        # Create vector database
        print("💾 Creating vector database from CSV...")
        self._create_vectordb()
        
        # Setup RAG chain
        print("⚙️  Setting up RAG chain...")
        self._setup_chain()
        
        print("\n" + "="*60)
        print("✅ System Ready!")
        print("="*60 + "\n")
    
    def _create_vectordb(self):
        """Create ChromaDB vector database from CSV data"""
        
        # Convert CSV rows to documents
        documents = []
        
        print(f"📝 Converting {len(self.df)} rows to documents...")
        
        for idx, row in self.df.iterrows():
            # Create text content from row
            content_parts = []
            for col in self.df.columns:
                content_parts.append(f"{col}: {row[col]}")
            
            text = " | ".join(content_parts)
            
            # Create Document object
            doc = Document(
                page_content=text,
                metadata={
                    "row_id": idx,
                    "source": self.csv_path
                }
            )
            documents.append(doc)
        
        print(f"✅ Created {len(documents)} documents\n")
        
        # Split documents into chunks
        print("✂️  Chunking documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=[" | ", ", ", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        print(f"✅ Created {len(splits)} chunks\n")
        
        # Create ChromaDB vector store
        print("🔄 Creating embeddings and storing in ChromaDB...")
        print("   (This may take a minute...)\n")
        
        # Delete old database if exists
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
        
        # Create new vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="csv_data"
        )
        
        print(f"✅ Vector database created with {len(splits)} chunks!\n")
    
    def _setup_chain(self):
        """Setup RAG chain with retriever"""
        
        # Create retriever from vector store
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Retrieve top 5 most similar chunks
        )
        
        # Create prompt template
        template = """You are a data analyst expert. Answer the question based ONLY on the following CSV data context.

CSV Data Context:
{context}

Question: {question}

Important Instructions:
- Use ONLY the information from the context above
- Be specific and accurate with numbers and values
- If the answer is not in the context, say "I don't have that information in the CSV data"
- Provide clear, direct answers

Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Helper function to format retrieved documents
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        # Create RAG chain using LCEL
        self.rag_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✅ RAG chain configured\n")
    
    def ask(self, question: str, show_sources: bool = False):
        """Ask a question about the CSV data"""
        
        print(f"\n{'─'*60}")
        print(f"❓ Question: {question}")
        print(f"{'─'*60}\n")
        
        print("🔍 Searching vector database...")
        
        # Get answer from RAG chain
        answer = self.rag_chain.invoke(question)
        
        print(f"\n💡 Answer:\n{answer}\n")
        
        # Optionally show source documents
        if show_sources:
            print(f"{'─'*60}")
            print("📚 Source Data Used:")
            print(f"{'─'*60}\n")
            
            # Retrieve relevant docs
            docs = self.retriever.invoke(question)
            
            for i, doc in enumerate(docs, 1):
                print(f"{i}. {doc.page_content[:200]}...")
                print(f"   [Row: {doc.metadata.get('row_id', 'N/A')}]\n")
        
        return answer
    
    def show_stats(self):
        """Show CSV statistics"""
        print("\n" + "="*60)
        print("📊 CSV Statistics")
        print("="*60)
        print(f"\nTotal Rows: {len(self.df)}")
        print(f"Total Columns: {len(self.df.columns)}")
        print(f"\nColumns: {', '.join(self.df.columns.tolist())}")
        print(f"\nVector DB Chunks: {self.vectorstore._collection.count()}")
        print()
    
    def chat(self):
        """Interactive chat mode"""
        
        print("="*60)
        print("💬 Interactive Chat - Ask questions about your CSV")
        print("="*60)
        print("\nCommands:")
        print("  'stats' - Show CSV statistics")
        print("  'sources' - Toggle showing sources (default: off)")
        print("  'quit' or 'exit' - Exit")
        print("="*60 + "\n")
        
        show_sources = False
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!\n")
                    break
                
                if question.lower() == 'stats':
                    self.show_stats()
                    continue
                
                if question.lower() == 'sources':
                    show_sources = not show_sources
                    print(f"\n{'✅' if show_sources else '❌'} Sources display: {'ON' if show_sources else 'OFF'}\n")
                    continue
                
                self.ask(question, show_sources=show_sources)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    """Main function"""
    
    print("\n" + "="*60)
    print("🚗 CSV Question Answering System")
    print("="*60 + "\n")
    
    # Get CSV file path
    csv_file = input("Enter CSV file name (default: cars.csv): ").strip()
    if not csv_file:
        csv_file = "cars.csv"
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"\n❌ Error: '{csv_file}' not found!\n")
        return
    
    try:
        # Initialize system
        rag = CSVRAGSystem(csv_file)
        
        # Show stats
        rag.show_stats()
        
        # Start chat
        rag.chat()
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()