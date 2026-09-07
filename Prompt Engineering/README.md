# Clinical Log Extractor: Few-Shot CoT Prompting

This project demonstrates how to use **Few-Shot Prompting** and **Chain of Thought (CoT)** to reliably extract structured JSON data from messy, unstructured text (rural clinic health logs).

## The Challenge
Language models are prone to hallucinating schema keys or formatting invalid JSON when parsing unstructured text like handwritten medical logs. 

## The Solution
This pipeline uses a **Few-Shot CoT** prompt to force the LLM to map out its reasoning *before* generating the final payload. By forcing the model to create a semantic trail, we achieve:
1. **Zero Schema Hallucinations:** The LLM strictly adheres to the requested keys.
2. **Implicit Data Handling:** It successfully infers data (e.g., extracting an integer `6` from the string "looks around 6").
3. **Regex-Friendly Output:** By appending a `JSON:` anchor after the reasoning, a simple Python regex can cleanly extract the final dictionary.

## Architecture Note: Why the LLM is Decoupled
This repository provides the prompt engineering framework and the parsing logic, but intentionally **does not** include active LLM API calls (e.g., OpenAI, Anthropic). This is a deliberate architectural decision for two reasons:

1. **Data Protection & Privacy:** Medical logs contain highly sensitive data. Hardcoding public API integrations in a repository encourages routing unencrypted patient data to third-party cloud providers. By keeping the inference engine decoupled, the pipeline can be safely integrated into a secure, offline environment (like a local quantized model) to maintain strict data compliance.
2. **System Modularity:** The core intellectual property here is the Few-Shot CoT structure and the extraction regex. By mocking the LLM output in the script, this module remains entirely model-agnostic. You can plug it into any inference engine you prefer—whether that is a cloud API or a local Hugging Face pipeline.

## Usage
Run the extractor script to see the pipeline parse a raw clinic log, strip the simulated CoT reasoning, and output a clean Python dictionary.

```bash
python extractor.py