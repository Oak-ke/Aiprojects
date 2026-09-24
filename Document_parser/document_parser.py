import os
import time
import requests
from google import genai
from google.genai import types
from google.genai.errors import APIError

def safe_generate_content(client, model, contents, config=None, retries=3, delay=5):
    """Helper function to retry API calls automatically if rate-limited or overloaded (429/503)"""
    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        except APIError as e:
            error_str = str(e)
            if any(code in error_str for code in ["429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]):
                if attempt < retries - 1:
                    print(f"\n⚠️ API limit/busy ({error_str[:30]}...). Retrying in {delay}s (Attempt {attempt + 1}/{retries})...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue
            raise e

if __name__ == "__main__":
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    # Image URL (.jpg)
    doc_url = "https://kileva.wordpress.com/wp-content/uploads/2012/02/scan-002.jpg?w=914"
    
    print(f"Downloading document from: {doc_url}...")
    response = requests.get(doc_url)
    response.raise_for_status()
    doc_bytes = response.content

    print("Document downloaded successfully. Sending to Gemini for layout-aware parsing...\n")

    parsing_prompt = """
    You are an advanced medical and document transcription engine. 
    Analyze this document, preserve its structural layout (headings, tables, or sections), 
    and extract all text cleanly. If there are any code-switched phrases (Swahili/English mix), 
    normalize them into clear English while retaining exact data fields.
    """

    # Use safe_generate_content with the correct 'image/jpeg' MIME type
    response = safe_generate_content(
        client=client,
        model="gemini-3.6-flash",
        contents=[
            types.Part.from_bytes(
                data=doc_bytes,
                mime_type="image/jpeg",  # Matches the .jpg URL
            ),
            parsing_prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.0
        )
    )

    print("--- PARSED DOCUMENT OUTPUT ---")
    print(response.text)