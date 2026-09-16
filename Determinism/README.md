# LLM Parameter Tuning: Temperature & Top-p

This project demonstrates how to control Large Language Model (LLM) **determinism** by manipulating probability parameters. Prompt engineering controls *what* the model processes, but parameter tuning controls *how* the model calculates its next-token generation.

## Core Concepts

### Determinism vs. Stochasticity
LLMs generate text by predicting the mathematical probability of the next word. 
* A **deterministic** model consistently chooses the highest-probability path. 
* A **stochastic** (creative) model is allowed to choose lower-probability paths, resulting in varied outputs.

### The Parameters
1. **Temperature (The Creativity Dial):** 
   * `0.0`: Forces the model to behave like a strict calculator, always picking the most likely next token. Essential for JSON extraction, coding, and medical triage.
   * `0.7 - 1.0`: Flattens the probability curve, encouraging the model to take risks. Ideal for marketing, brainstorming, and storytelling.
2. **Top-p / Nucleus Sampling (The Vocabulary Pool):**
   * `0.1`: The model only considers words that make up the top 10% of cumulative probability, resulting in highly restricted, predictable vocabulary.
   * `0.9`: Considers a massive pool of words, allowing for diverse and engaging language.

## The Practical Application
This script initializes two distinct LLM configurations to handle the exact same clinical scenario (a new rural clinic opening):
* **`triage_llm`:** Configured with `temp=0.0` and `top_p=0.1`. Generates a rigid, protocol-driven internal memo. If run 10 times, the output will remain nearly identical.
* **`marketing_llm`:** Configured with `temp=0.8` and `top_p=0.9`. Generates warm, engaging community outreach SMS copy. Every execution yields a uniquely phrased message.

## Usage

### 1. Set API Credentials
Export your Google Gemini API key securely in your terminal:

**Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY="your_api_key_here"
```

### 2. Run the Comparison
Execute the script to observe the contrast between the deterministic and stochastic models in real-time

```bash
python determinism_test.py
```
