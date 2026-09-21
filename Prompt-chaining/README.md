# Prompt Chaining Pipeline for Code-Switched Healthcare Data

This project demonstrates an advanced **Prompt Chaining Pipeline** designed to solve a common challenge in East African healthcare informatics: **linguistic code-switching**. 

## The Business & Operational Challenge
In community health settings across Kenya, patient intake logs and clinical notes rarely follow rigid textbook English. They often feature a fluid mix of English, Swahili, Sheng, and regional vernacular (e.g., *"MgONJWA ana chest pains kali sana, ame-suffer for 3 days, alipewa Panadol lakini hakuna improvement"*). 

Feeding raw, unstructured, multi-lingual text directly into a database parser or diagnostic support tool leads to parsing errors, missed symptoms, and data corruption. 

## The Solution: Two-Stage Pipeline Architecture
Instead of burdening a single prompt with translation, linguistic normalization, and JSON formatting simultaneously, this project splits the workflow into a resilient two-stage pipeline:

1. **Stage 1 (Normalization & Translation):** 
   - Takes raw, code-switched clinical logs.
   - Translates and standardizes them into clean, professional English medical notes while preserving exact symptom details, durations, and medications.
2. **Stage 2 (Structured Parsing & Schema Enforcement):** 
   - Takes the normalized English text from Stage 1.
   - Uses **Pydantic** combined with native schema enforcement (`response_schema`) to output a 100% type-safe, validated JSON object ready for database ingestion.

## Tech Stack
* **Language:** Python
* **LLM Client:** Native Google GenAI SDK (`google-genai`)
* **Data Validation:** Pydantic (`BaseModel`)
* **Core Techniques:** Prompt Chaining, Schema Enforcement, Code-Switching Normalization

## Project Structure
```text
├── clinic_pipeline.py      # Main two-stage pipeline execution script
├── requirements.txt        # Project dependencies (google-genai, pydantic)
└── README.md               # Project documentation
```

## Setup & Usage
1. Install Dependencies
Ensure you have your virtual environment activated, then install the required packages:

``` bash
pip install google-genai pydantic
```

## 2. Set API Credentials
Export your Google Gemini API key securely in your terminal:

Windows (PowerShell):

```PowerShell
$env:GOOGLE_API_KEY="your_api_key_here"
```

## 3. Run the Pipeline
```Bash
python clinic_pipeline.py
```