import os
from pydantic import BaseModel, Field
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. Define the Pydantic Schema (The Blueprint) ---
class MicroLoanExtraction(BaseModel):
    borrower_name: str = Field(
        description="The name of the person borrowing the money."
    )
    principal_amount: int = Field(
        description="The original amount borrowed, formatted as an integer."
    )
    total_repayment: int = Field(
        description="The total amount to be paid back. If not stated, assume it equals the principal."
    )
    due_date: str = Field(
        description="When the loan is expected to be repaid."
    )
    purpose: Optional[str] = Field(
        default=None, 
        description="What the loan is being used for, if stated."
    )

if __name__ == "__main__":
    # --- 2. Initialize the API Connection (The Waiter) ---
    # Make sure you set your API key in your terminal before running this script!
    # Windows: $env:GOOGLE_API_KEY="your-key-here"
    # Mac/Linux: export GOOGLE_API_KEY="your-key-here"
    
    # We use gemini-1.5-flash because it is lightning fast and perfect for data extraction
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        temperature=0, # 0 means strictly adhere to the facts, no creativity
        api_key=os.environ.get("GOOGLE_API_KEY") 
    )

    # --- 3. Bind the Schema to the LLM ---
    # This tells the API: "You are physically not allowed to output anything except this exact Pydantic format."
    structured_llm = llm.with_structured_output(MicroLoanExtraction)

    # --- 4. Execute with unstructured text ---
    raw_sms_log = "Sent 1500 to Omondi for his boda repair. He promised to send back 1700 by end of the month."
    
    print("Sending unstructured SMS to the API...")
    
    # .invoke() sends the text to the API and waits for the response
    extracted_data = structured_llm.invoke(raw_sms_log)

    # --- 5. View the Results ---
    # Because of Pydantic, 'extracted_data' is instantly a valid Python object. 
    # Notice we can do math on the integers right away without getting an error!
    print("\n--- Extraction Successful ---")
    print(f"Borrower: {extracted_data.borrower_name}")
    print(f"Principal: {extracted_data.principal_amount} KES")
    print(f"Total Repayment: {extracted_data.total_repayment} KES")
    print(f"Profit (Interest): {extracted_data.total_repayment - extracted_data.principal_amount} KES")
    print(f"Due: {extracted_data.due_date}")
    print(f"Purpose: {extracted_data.purpose}")