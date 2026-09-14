# Micro-Loan Data Extractor: Structured Outputs with Pydantic

This project demonstrates how to use **Large Language Models (LLMs)** combined with **Pydantic** to reliably extract structured financial data from informal, unstructured text (like SMS or WhatsApp ledgers) common in rural micro-finance.

## The Challenge
In informal economies, loan data is often communicated via unstructured messages (e.g., *"Sent 1500 to Omondi for his boda repair"*). Traditional extraction methods like Regular Expressions (Regex) are brittle and fail when sentence structure changes or numbers are spelled out as words. Unconstrained LLM outputs are equally risky, as they can hallucinate JSON schema keys or output incorrect data types.

## The Solution: Type-Safe Schema Enforcement
This pipeline abandons Regex and uses **LangChain's Structured Outputs** to bind a strict Pydantic model directly to the LLM generation process. 

By passing the `MicroLoanExtraction` schema to the API:
1. **Guaranteed Structure:** The model is physically constrained to output the exact keys required by the database.
2. **Strict Type Safety:** Fields like `principal_amount` are guaranteed to be integers, allowing immediate mathematical operations (like calculating interest) without casting strings.
3. **Graceful Degradation:** Using `Optional[str]` allows the pipeline to safely handle missing data (like a missing loan purpose) without crashing.

## Architecture & Tech Stack
* **Language:** Python
* **Validation:** Pydantic (`BaseModel`, `Field`)
* **Orchestration:** LangChain (`langchain-google-genai`)
* **Inference:** Google Gemini 1.5 Flash (via API)

## Usage

### 1. Install Dependencies
```bash
pip install langchain-google-genai pydantic
```

### 2. Windows Powershell
```bash 
$env:GOOGLE_API_KEY="your_api_key_here"
```

### 3. Run the Extractor
``` bash
python extract_loans.py
```




