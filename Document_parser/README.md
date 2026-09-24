# Layout-Aware Multimodal Document Parser for Clinical Records

This project explores **Multimodal Document Parsing**, demonstrating how modern Vision-Language Models surpass traditional **OCR (Optical Character Recognition)** by preserving structural layouts, tables, and hierarchical context while handling multilingual text.

## The Challenge: Legacy OCR in Healthcare
Traditional OCR systems scan documents pixel-by-pixel, outputting flat walls of text with zero regard for spatial layout. In clinical environments across East Africa—where patient intake registers, referral forms, and health logs are often scanned or handwritten—legacy OCR completely fails:
* It scrambles table columns and form fields.
* It chokes on heavy **linguistic code-switching** (a fluid mix of English medical terms, Swahili, and local Sheng).
* It loses critical contextual markers found in headings or margins.

## The Solution: Layout-Aware Multimodal Parsing
By using Google Gemini’s multimodal document processing capabilities via the native SDK, this pipeline reads documents much like a human clinician does. It inspects the visual layout of the PDF, maps data fields accurately, and normalizes code-switched notes into clean, structured text ready for downstream databases.

## Tech Stack
* **Language:** Python
* **LLM Client:** Native Google GenAI SDK (`google-genai`)
* **Network Handling:** `requests` for remote document retrieval
* **Core Techniques:** Multimodal byte streaming, Layout-Aware Extraction, Code-Switching Normalization

## Project Structure
```text
├── document_parser.py      # Main layout-aware parsing script
├── requirements.txt        # Project dependencies (google-genai, requests)
└── README.md               # Project documentation
```

## Setup & Usage
### 1. Install Dependencies
Ensure your virtual environment is active, then install the required packages:

``` Bash
pip install google-genai requests
``` 

## 2. Set API Credentials
### 3. Export your Google Gemini API key securely in your terminal:

Windows (PowerShell):

```PowerShell
$env:GOOGLE_API_KEY="your_api_key_here"

```

### 3. Run the Parser
```Bash
python document_parser.py
```