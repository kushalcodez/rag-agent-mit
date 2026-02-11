from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("loanapproval.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chroma_loan_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        # Create a text representation of each loan application
        page_content = f"""
        Applicant {row['applicant_id']}: {row['age']} year old {row['gender']}, {row['marital_status']}.
        Employment: {row['employment_status']}.
        Annual Income: ${row['annual_income']}.
        Credit Score: {row['credit_score']}.
        Loan Amount Requested: ${row['loan_amount']}.
        Number of Dependents: {row['num_dependents']}.
        Existing Loans: {row['existing_loans_count']}.
        Loan Status: {'Approved' if row['loan_approved'] == 1 else 'Rejected'}.
        """
        
        document = Document(
            page_content=page_content.strip(),
            metadata={
                "applicant_id": row["applicant_id"],
                "age": row["age"],
                "gender": row["gender"],
                "marital_status": row["marital_status"],
                "annual_income": row["annual_income"],
                "loan_amount": row["loan_amount"],
                "credit_score": row["credit_score"],
                "num_dependents": row["num_dependents"],
                "existing_loans_count": row["existing_loans_count"],
                "employment_status": row["employment_status"],
                "loan_approved": row["loan_approved"]
            },
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="loan_applications",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
# Increased k from 5 to 50 for better statistical coverage
retriever = vector_store.as_retriever(
    search_kwargs={"k": 50}
)