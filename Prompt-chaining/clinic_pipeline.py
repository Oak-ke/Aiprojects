import os
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field

class ClinicalExtraction(BaseModel):
    primary_symptoms: list[str] = Field(description="Primary symptoms identified.")
    symptom_duration: str = Field(description="Duration of symptoms.")
    previous_interventions: list[str] = Field(description="Medications attempted.")
    severity_indicators: str = Field(description="Severity notes.")

def safe_generate_content(client, model, contents, config=None, retries=3, delay=5):
    """Helper function to retry API calls automatically if rate-limited (429)"""
    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        except APIError as e:
            if "429" in str(e) or "503" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "UNAVAILABLE" in str(e):
                if attempt < retries - 1:
                    print(f"\n Rate limit hit (429). Retrying in {delay} seconds (Attempt {attempt + 1}/{retries})...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff: double the wait time each retry
                    continue
            raise e

if __name__ == "__main__":
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    raw_clinic_log = (
        "MgONJWA ana homa kali sana na chest pains kwa 3 days. "
        "Alipewa Panadol kwk duka lakini hakuna improvement, anakohoa sana usiku."
    )

    print("--- RAW INPUT ---")
    print(raw_clinic_log + "\n")

    # ==========================================
    # STAGE 1: Translation & Normalization
    # ==========================================
    translator_prompt = f"""
    You are a medical data transcriptionist in East Africa. 
    Translate and normalize the following code-switched patient log into clear, professional, standardized English medical notes. 
    Preserve all specific symptoms, durations, and medications mentioned. Do not add outside information.

    Raw Log:
    {raw_clinic_log}
    """

    print("Running Stage 1: Translating and normalizing...")
    stage_1_response = safe_generate_content(
        client=client,
        model="gemini-flash-latest",
        contents=translator_prompt
    )
    
    normalized_notes = stage_1_response.text
    print("\n--- STAGE 1 OUTPUT (Normalized English) ---")
    print(normalized_notes)

    # Buffer pause between pipeline steps
    time.sleep(2)

    # ==========================================
    # STAGE 2: Structured Parsing with Pydantic
    # ==========================================
    parser_prompt = f"""
    Analyze the standardized medical notes below and extract the clinical details according to the required schema format.

    Standardized Medical Notes:
    {normalized_notes}
    """

    print("\nRunning Stage 2: Parsing into Pydantic JSON schema...")
    
    stage_2_response = safe_generate_content(
        client=client,
        model="gemini-flash-latest",
        contents=parser_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ClinicalExtraction,
            temperature=0.0
        )
    )

    print("\n--- STAGE 2 OUTPUT (Validated Pydantic Object / JSON) ---")
    print(stage_2_response.text)
    
    parsed_data = ClinicalExtraction.model_validate_json(stage_2_response.text)
    print("\nPython Object Type Check:", type(parsed_data))
    print("Extracted Symptoms List:", parsed_data.primary_symptoms)