from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import pandas as pd

model = OllamaLLM(model="llama3.2")

# Load the full dataset for statistical queries
df = pd.read_csv("loanapproval.csv")

template = """
You are an expert in analyzing loan application data.

Here is statistical information from the complete database:
{statistics}

Here are some relevant loan applications as examples: {applications}

Based on this data, answer the following question: {question}

Use the statistical information for accurate numbers and the example applications to illustrate patterns.
Be specific and data-driven in your answers.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def get_statistics(question):
    """Generate relevant statistics based on the question"""
    stats = []
    
    # Basic stats
    stats.append(f"Total Applications: {len(df)}")
    stats.append(f"Approved: {df['loan_approved'].sum()} ({df['loan_approved'].mean()*100:.1f}%)")
    stats.append(f"Rejected: {(1-df['loan_approved']).sum()} ({(1-df['loan_approved'].mean())*100:.1f}%)")
    
    # Employment status breakdown
    if 'employment' in question.lower() or 'unemployed' in question.lower():
        stats.append("\nEmployment Status Breakdown:")
        for status in df['employment_status'].unique():
            subset = df[df['employment_status'] == status]
            approval_rate = subset['loan_approved'].mean() * 100
            stats.append(f"  {status}: {len(subset)} applications, {approval_rate:.1f}% approved")
    
    # Credit score stats
    if 'credit' in question.lower():
        stats.append(f"\nCredit Score Statistics:")
        stats.append(f"  Overall Average: {df['credit_score'].mean():.0f}")
        stats.append(f"  Approved Average: {df[df['loan_approved']==1]['credit_score'].mean():.0f}")
        stats.append(f"  Rejected Average: {df[df['loan_approved']==0]['credit_score'].mean():.0f}")
    
    # Income stats
    if 'income' in question.lower():
        stats.append(f"\nIncome Statistics:")
        stats.append(f"  Overall Average: ${df['annual_income'].mean():,.0f}")
        stats.append(f"  Approved Average: ${df[df['loan_approved']==1]['annual_income'].mean():,.0f}")
        stats.append(f"  Rejected Average: ${df[df['loan_approved']==0]['annual_income'].mean():,.0f}")
    
    # Age stats
    if 'age' in question.lower():
        stats.append(f"\nAge Statistics:")
        stats.append(f"  Overall Average: {df['age'].mean():.1f} years")
        stats.append(f"  Approved Average: {df[df['loan_approved']==1]['age'].mean():.1f} years")
        stats.append(f"  Rejected Average: {df[df['loan_approved']==0]['age'].mean():.1f} years")
    
    # Marital status
    if 'marital' in question.lower() or 'married' in question.lower():
        stats.append("\nMarital Status Breakdown:")
        for status in df['marital_status'].unique():
            subset = df[df['marital_status'] == status]
            approval_rate = subset['loan_approved'].mean() * 100
            stats.append(f"  {status}: {len(subset)} applications, {approval_rate:.1f}% approved")
    
    # Gender breakdown
    if 'gender' in question.lower() or 'male' in question.lower() or 'female' in question.lower():
        stats.append("\nGender Breakdown:")
        for gender in df['gender'].unique():
            subset = df[df['gender'] == gender]
            approval_rate = subset['loan_approved'].mean() * 100
            stats.append(f"  {gender}: {len(subset)} applications, {approval_rate:.1f}% approved")
    
    return "\n".join(stats)

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question about loan applications (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    # Get relevant statistics
    statistics = get_statistics(question)
    
    # Get example applications
    applications = retriever.invoke(question)
    
    result = chain.invoke({
        "statistics": statistics,
        "applications": applications,
        "question": question
    })
    print(result)