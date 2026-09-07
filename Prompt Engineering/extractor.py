import re
import json

# 1. Define the Few-Shot CoT Prompt Template
PROMPT_TEMPLATE = """You are a medical data extraction system. Your role is to extract structured JSON from unstructured rural health logs.

Follow this exact process:
1. REASONING: Step-by-step, identify the patient name, age, symptoms, diagnosis, and treatment plan. If a field is missing, state that it is null.
2. JSON: Output ONLY valid JSON containing the keys: "patient_name", "age", "symptoms" (array), "diagnosis", "treatment", "follow_up_required" (boolean).

--- Example 1 ---
Log: "0800hrs - Brought in baby Wanjiku, looks about 2 yrs old. Mother says high fever and sweating since Tuesday. RDT positive. Gave AL syrup and panadol."

REASONING:
- Name: Identified directly as "Wanjiku".
- Age: Inferred as 2 from "looks about 2 yrs old".
- Symptoms: The mother reported "high fever and sweating".
- Diagnosis: The log states "RDT positive", which implies a diagnosis of Malaria.
- Treatment: Given "AL syrup and panadol".
- Follow-up: Not explicitly stated in the text, so this will be false.

JSON:
{
  "patient_name": "Wanjiku",
  "age": 2,
  "symptoms": ["high fever", "sweating"],
  "diagnosis": "Malaria",
  "treatment": "AL syrup, panadol",
  "follow_up_required": false
}

--- Task ---
Log: "{clinic_log}"

REASONING:
"""

def extract_json_payload(llm_response: str) -> dict:
    """
    Parses the LLM's raw text response, strips away the CoT reasoning, 
    and returns the structured JSON payload.
    """
    # Regex to capture everything between the first { and the last }
    json_match = re.search(r'(\{.*\})', llm_response, re.DOTALL)
    
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            return {"error": "LLM generated invalid JSON formatting."}
    return {"error": "No JSON payload found in the response."}

# --- Test the Pipeline ---
if __name__ == "__main__":
    new_log = "14/09 morning shift. Brought in young boy, looks around 6. Name is Kiprono. Mother reports severe stomach cramps and vomiting since last night. Temp normal. Suspect food poisoning from bad water. Gave ORS packets and advised rest."
    
    # In a real app, you would pass `final_prompt` to your LLM here.
    final_prompt = PROMPT_TEMPLATE.format(clinic_log=new_log)
    
    # Mocking the LLM's response for testing
    mock_llm_output = """- Name: Identified directly as "Kiprono".
- Age: Inferred as 6 from "looks around 6".
- Symptoms: The mother reported "severe stomach cramps and vomiting".
- Diagnosis: The text states "Suspect food poisoning", which serves as the diagnosis.
- Treatment: "Gave ORS packets and advised rest".
- Follow-up: Not explicitly stated, so this will be false.

JSON:
{
  "patient_name": "Kiprono",
  "age": 6,
  "symptoms": ["severe stomach cramps", "vomiting"],
  "diagnosis": "food poisoning",
  "treatment": "ORS packets and advised rest",
  "follow_up_required": false
}"""

    # Extract and print the final JSON
    parsed_data = extract_json_payload(mock_llm_output)
    print("--- Extracted Python Dictionary ---")
    print(json.dumps(parsed_data, indent=2))