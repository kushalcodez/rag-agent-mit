import os
from typing import List
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document

class CSVRAGAgent:
    def __init__(self, 
                 model_name="llama3.2:3b",
                 embedding_model="mxbai-embed-large",
                 persist_directory="./chroma_db"):
        """
        Initialize the RAG agent for CSV files
        
        Args:
            model_name: The Ollama LLM model to use
            embedding_model: The Ollama embedding model (mxbai)
            persist_directory: Where to save the vector database
        """
        print("Initializing CSV RAG Agent...")
        
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        
        # Initialize embeddings using mxbai
        print("Loading embedding model...")
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Initialize LLM
        print("Loading LLM...")
        self.llm = OllamaLLM(model=model_name, temperature=0)
        
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        
    def load_csv_files(self, file_paths: List[str]) -> List[Document]:
        """
        Load CSV files
        
        Args:
            file_paths: List of CSV file paths
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for file_path in file_paths:
            if not file_path.endswith('.csv'):
                print(f"Skipping non-CSV file: {file_path}")
                continue
                
            print(f"Loading CSV: {file_path}")
            try:
                loader = CSVLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                print(f"  Loaded {len(docs)} rows from {file_path}")
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
        
        print(f"\nTotal loaded: {len(documents)} rows from CSV files")
        return documents
    
    def load_from_folder(self, folder_path: str) -> List[str]:
        """
        Get all CSV files from a folder
        
        Args:
            folder_path: Path to folder containing CSV files
            
        Returns:
            List of CSV file paths
        """
        csv_files = []
        
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return csv_files
        
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                csv_files.append(os.path.join(folder_path, file))
        
        print(f"Found {len(csv_files)} CSV files in {folder_path}")
        return csv_files
    
    def process_documents(self, documents: List[Document], 
                         chunk_size=1000, 
                         chunk_overlap=200) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of Document objects
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            
        Returns:
            List of chunked documents
        """
        print(f"\nProcessing documents into chunks...")
        print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ", ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} rows")
        return chunks
    
    def create_vectorstore(self, chunks: List[Document]):
        """
        Create vector database from document chunks
        
        Args:
            chunks: List of document chunks
        """
        print("\nCreating vector store...")
        print("(This may take a while depending on the number of chunks)")
        
        if self.vectorstore is None:
            # Create new vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print("Vector store created successfully!")
        else:
            # Add to existing vectorstore
            self.vectorstore.add_documents(chunks)
            print("Documents added to existing vector store!")
    
    def load_existing_vectorstore(self):
        """
        Load an existing vector database from disk
        """
        if os.path.exists(self.persist_directory):
            print(f"\nLoading existing vector store from {self.persist_directory}...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("Vector store loaded successfully!")
            return True
        else:
            print(f"No existing vector store found at {self.persist_directory}")
            return False
    
    def setup_chain(self, k=4):
        """
        Setup the question-answering chain using LCEL
        
        Args:
            k: Number of relevant chunks to retrieve
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Load documents first or load existing vectorstore.")
        
        print(f"\nSetting up QA chain (retrieving top {k} chunks)...")
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        
        # Create prompt template
        template = """You are a helpful assistant that answers questions based on CSV data.
Use the following context from CSV files to answer the question. If you cannot find the answer in the context, say so.

Context from CSV files:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the chain using LCEL
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("QA chain ready!")
    
    def ask(self, question: str, show_sources=True):
        """
        Ask a question about your CSV data
        
        Args:
            question: The question to ask
            show_sources: Whether to display source documents
            
        Returns:
            Answer string
        """
        if self.chain is None:
            raise ValueError("Chain not initialized. Run setup_chain() first.")
        
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        # Get the answer
        answer = self.chain.invoke(question)
        
        print(f"\nAnswer:\n{answer}")
        
        # Optionally show source documents
        if show_sources and self.retriever:
            print(f"\n{'-'*60}")
            print("Sources:")
            print(f"{'-'*60}")
            
            # Retrieve the source documents
            source_docs = self.retriever.get_relevant_documents(question)
            
            for i, doc in enumerate(source_docs, 1):
                print(f"\nSource {i}:")
                print(f"Content: {doc.page_content[:300]}...")
                if 'source' in doc.metadata:
                    print(f"File: {doc.metadata['source']}")
                if 'row' in doc.metadata:
                    print(f"Row: {doc.metadata['row']}")
        
        return answer
    
    def interactive_mode(self):
        """
        Start interactive question-answering mode
        """
        print("\n" + "="*60)
        print("Interactive Mode - Ask questions about your CSV data")
        print("Type 'quit', 'exit', or 'q' to stop")
        print("="*60 + "\n")
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q', '']:
                    print("Goodbye!")
                    break
                
                self.ask(question)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """
    Main function demonstrating how to use the CSV RAG Agent
    """
    
    # Step 1: Initialize the agent
    print("="*60)
    print("CSV RAG Agent - Question Answering System")
    print("="*60)
    
    agent = CSVRAGAgent(
        model_name="llama3.2:3b",
        embedding_model="mxbai-embed-large",
        persist_directory="./csv_chroma_db"
    )
    
    # Step 2: Check if vector store already exists
    if agent.load_existing_vectorstore():
        # Vector store exists, just set up the chain
        agent.setup_chain(k=4)
    else:
        # Need to load and process documents
        
        # Option A: Load specific CSV files
        csv_files = [
            "cars.csv"
        ]
        
        # Option B: Load all CSVs from a folder
        # csv_files = agent.load_from_folder("./csv_data")
        
        # Load the CSV files
        documents = agent.load_csv_files(csv_files)
        
        if len(documents) == 0:
            print("No documents loaded! Please check your file paths.")
            return
        
        # Process into chunks
        chunks = agent.process_documents(
            documents,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Create vector store
        agent.create_vectorstore(chunks)
        
        # Setup QA chain
        agent.setup_chain(k=4)
    
    # Step 3: Ask questions
    print("\n" + "="*60)
    print("Ready to answer questions!")
    print("="*60)
    
    # Example questions
    agent.ask("What is the total revenue in the data?")
    agent.ask("Who are the top 5 customers?")
    agent.ask("What is the average order value?")
    
    # Step 4: Interactive mode (optional)
    agent.interactive_mode()


if __name__ == "__main__":
    main()