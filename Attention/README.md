# M-PESA Sentiment Analysis with Self-Attention

## What It Does

This model classifies M-PESA transaction messages as **Successful** or **Failed** using a self-attention mechanism. It analyzes the text of M-PESA messages and identifies key patterns like "received", "confirmed" (success) or "failed", "declined", "insufficient" (failure).

## How It Works

1. Converts text messages to numerical tokens
2. Processes through a self-attention layer to understand context
3. Classifies messages as Successful (1) or Failed (0)
4. Visualizes which words influenced the decision

## Dependencies

- Python 3.8+
- PyTorch

## Installation

```bash
pip install torch

## How to Run

1. Save the code as mpesa_sentiment.py
2. Run the script

```bash
python mpesa_sentiment.py

3. Expected Output
The script will:

Train on sample M-PESA messages

Show training progress (loss and accuracy)

Test on new messages with confidence scores

Display attention visualization showing important words