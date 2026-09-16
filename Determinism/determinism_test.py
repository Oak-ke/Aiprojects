import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# --- 1. The Strict Medical Model (Determinism = High) ---
# Temp 0.0 ensures protocol adherence. No hallucination allowed.
triage_llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.0,
    top_p=0.1, # Extremely restricted vocabulary pool
    api_key=os.environ.get("GOOGLE_API_KEY"),
    max_retries=0,
    timeout=15
)

# --- 2. The Creative Marketing Model (Determinism = Low) ---
# Temp 0.8 allows for engaging, persuasive, and varied language.
marketing_llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.8,
    top_p=0.9, # Wide vocabulary pool
    api_key=os.environ.get("GOOGLE_API_KEY"),
     max_retries=0,
    timeout=15
)

# --- 3. The Test Data ---
clinical_scenario = "We are opening a new rural clinic in Bungoma offering free malaria testing and basic maternal care."

triage_prompt = f"Write a strict, 2-sentence internal protocol for handling patients arriving at this facility: {clinical_scenario}"
marketing_prompt = f"Write a warm, engaging 2-sentence SMS broadcast to send to the community about this facility: {clinical_scenario}"

if __name__ == "__main__":
    print("Generating Strict Triage Protocol (Temp 0.0)...")
    
    # .stream() yields chunks of text in real-time as they are generated
    for chunk in triage_llm.stream([HumanMessage(content=triage_prompt)]):
        print(chunk.content, end="", flush=True)
    print("\n")

    print("\nGenerating Creative Marketing SMS (Temp 0.8)...")
    
    for chunk in marketing_llm.stream([HumanMessage(content=marketing_prompt)]):
        print(chunk.content, end="", flush=True)
    print("\n")