# AI Safety in Healthcare: CHW Triage Assistant

This repository demonstrates advanced prompt engineering techniques—specifically **Persona Definition** and **Hard Bounds**—designed to control Large Language Model (LLM) behavior in high-stakes environments.

## The Objective
To design a triage assistant for Community Health Workers (CHWs) that provides structured, accessible medical support without crossing the line into unauthorized medical diagnosis or prescription.

## Core Engineering Concepts

### 1. The Persona (`[ROLE AND PERSONA]`)
The LLM is constrained to act as an assistant to a field worker, not a physician. The vocabulary is explicitly restricted to plain language, ensuring the output is immediately actionable by CHWs without requiring advanced medical training.

### 2. Hard Bounds (`[BOUNDS AND CONSTRAINTS]`)
To ensure medical safety, the prompt establishes unbreakable negative constraints:
*   **Protocol Enforcement:** The model is forbidden from diagnosing or prescribing non-standard medications.
*   **Emergency Hijack:** If the model detects specific high-risk concepts (e.g., "unable to drink", "unconscious"), it abandons standard formatting and triggers a hard-coded emergency escalation.

## Testing the Bounds
The prompt successfully intercepts critical inputs. 
*   **Input:** "Child is severely dehydrated and unable to drink water."
*   **Output:** ` EMERGENCY ESCALATION REQUIRED. REFER TO NEAREST CLINIC IMMEDIATELY.`

## Use Case
This prompt architecture is designed for integration into offline-first or low-bandwidth systems (e.g., local RAG pipelines) deployed in rural or resource-constrained settings where strict adherence to medical protocols is critical.