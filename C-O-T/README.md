# Chain of Thought (CoT) Agricultural Logistics Calculator

This project demonstrates how to use **Chain of Thought (CoT) prompting** to guide Large Language Models through multi-step arithmetic and business logic. By forcing the model to show its reasoning trace sequentially, we eliminate mathematical hallucinations—making LLMs reliable for financial and operational planning.

## The Business Challenge
In agricultural mechanization services (such as tractor deployment for seasonal ploughing and harrowing), job pricing is rarely a single flat fee. It depends on multiple interacting variables:
* Acreage size
* Distance from the machinery depot (introducing transport surcharges)
* Fixed operator allowances

When unconstrained LLMs are asked to compute these totals in a single step, they frequently make basic multiplication or addition errors. 

## The Solution: Chain of Thought (CoT)
Instead of asking the model for a final number, the CoT prompt structures the generation into strict, sequential steps:
1. **Step 1:** Calculate baseline tillage costs.
2. **Step 2:** Evaluate distance rules against threshold policies.
3. **Step 3:** Account for fixed operational allowances.
4. **Step 4:** Aggregate the final sum.

This sequential anchoring forces the model's token prediction to follow human-like arithmetic logic.

## Architecture & Tech Stack
* **Language:** Python
* **Orchestration:** LangChain (`langchain-google-genai`)
* **Inference Model:** Google Gemini Flash
* **Core Technique:** Step-by-Step Chain of Thought (CoT) & Determinism Control

## Usage

### 1. Set API Credentials
Export your Google Gemini API key securely in your terminal:

**Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY="your_api_key_here"