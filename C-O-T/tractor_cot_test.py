import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Initialize the model (using a standard flash model for quick execution)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.0, # 0.0 ensures the model follows the logic strictly without straying
    api_key=os.environ.get("GOOGLE_API_KEY")
)

# Define the scenario parameters
acreage = 12
distance_km = 25

# The Chain of Thought Prompt Template
cot_prompt = f"""
[SYSTEM ROLE]
You are an expert agricultural logistics and financial planning assistant for farm mechanization services in Western/Rift Valley Kenya. Your job is to calculate accurate deployment costs by breaking down the math step-by-step.

[INSTRUCTIONS & RULES]
1. DO NOT jump straight to the final answer. You must use a Step-by-Step Chain of Thought (CoT).
2. Base tractor tillage operations strictly on a flat rate of 4,000 KES per acre.
3. If distance exceeds 20 km from the base depot, add a flat transport surcharge of 1,500 KES.
4. Operator field allowance is fixed at a flat 500 KES per job.

--- Farm Job Request ---
Acreage to till: {acreage} acres
Distance from depot: {distance_km} km

[REQUIRED OUTPUT FORMAT]
**Step 1 - Base Tillage Calculation:** (Show math: acreage * rate)
**Step 2 - Distance & Surcharge Check:** (Evaluate distance against the 20km threshold)
**Step 3 - Additional Allowances:** (List operator allowance)
**Step 4 - Final Total Cost:** (Sum all items clearly)
"""

if __name__ == "__main__":
    print(f"Calculating logistics for a {acreage}-acre farm, {distance_km} km away...\n")
    
    # Execute the model with streaming to watch the reasoning trace unfold
    for chunk in llm.stream([HumanMessage(content=cot_prompt)]):
        print(chunk.content, end="", flush=True)
    print("\n")