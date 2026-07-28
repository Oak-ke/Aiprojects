import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
from collections import Counter

# ---------- MODEL DEFINITION ----------
class MPESASentimentAttention(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        embeddings = self.embedding(x)
        Q = self.W_q(embeddings)
        K = self.W_k(embeddings)
        V = self.W_v(embeddings)
        
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        summary_vector = torch.mean(context, dim=1)
        logits = self.classifier(summary_vector)
        return logits, attention_weights

# ---------- TEXT PROCESSING ----------
def create_vocabulary(texts, vocab_size=100):
    """Create vocabulary from M-PESA messages"""
    # Collect all words
    all_words = []
    for text in texts:
        words = text.lower().split()
        all_words.extend(words)
    
    # Get most common words
    word_counts = Counter(all_words)
    most_common = word_counts.most_common(vocab_size - 2)  # Reserve for <PAD> and <UNK>
    
    # Create word to index mapping
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1
    }
    for idx, (word, _) in enumerate(most_common, start=2):
        vocab[word] = idx
    
    return vocab

def text_to_tensor(text, vocab, max_len=20):
    """Convert text to tensor of token IDs"""
    words = text.lower().split()[:max_len]
    token_ids = [vocab.get(word, vocab['<UNK>']) for word in words]
    
    # Pad to max_len
    while len(token_ids) < max_len:
        token_ids.append(vocab['<PAD>'])
    
    return torch.tensor(token_ids)

# ---------- MAIN TEST ----------
print("=" * 60)
print("M-PESA SENTIMENT ANALYSIS")
print("=" * 60)

# 1. Sample M-PESA messages with labels
# 1 = Good/Successful, 0 = Bad/Failed
mpesa_messages = [
    # Successful transactions (label = 1)
    "M-PESA confirmed you have received 5000 KES from John",
    "Your payment of 1500 to Jane was successful",
    "M-PESA: You have received 2000 from Peter",
    "Transaction successful, balance is 12000",
    "M-PESA confirmed withdrawal of 3000 from your account",
    "You have received 1000 from Mercy",
    "Payment of 2500 completed successfully",
    "M-PESA: Your account has been credited with 5000",
    
    # Failed transactions (label = 0)
    "M-PESA transaction failed due to insufficient funds",
    "Your transaction has been declined",
    "M-PESA: Failed to process your request",
    "Transaction failed, please try again",
    "Insufficient balance for this transaction",
    "The transaction was unsuccessful",
    "M-PESA: Your payment has been rejected",
    "Transaction could not be completed",
]

# Labels: 1 = successful, 0 = failed
labels = [
    1, 1, 1, 1, 1, 1, 1, 1,  # Successful
    0, 0, 0, 0, 0, 0, 0, 0   # Failed
]

print(f"\nDataset: {len(mpesa_messages)} M-PESA messages")
print(f"   Successful: {sum(labels)}")
print(f"   Failed: {len(labels) - sum(labels)}")

# 2. Create vocabulary
vocab = create_vocabulary(mpesa_messages, vocab_size=50)
print(f"\nVocabulary size: {len(vocab)} words")
print(f"   Sample words: {list(vocab.keys())[:10]}")

# 3. Convert texts to tensors
max_len = 15
X = torch.stack([text_to_tensor(msg, vocab, max_len) for msg in mpesa_messages])
y = torch.tensor(labels)

print(f"\nInput tensor shape: {X.shape}")
print(f"   Labels: {y.tolist()}")

# 4. Create model
model = MPESASentimentAttention(
    vocab_size=len(vocab),
    d_model=32,
    num_classes=2
)

print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

# 5. Train the model
print("\nTraining model on M-PESA messages...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

model.train()
for epoch in range(100):
    # Shuffle data
    indices = torch.randperm(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Forward pass
    logits, _ = model(X_shuffled)
    loss = loss_fn(logits, y_shuffled)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_shuffled).float().mean()
        print(f"   Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, Accuracy = {acc.item():.2%}")

# 6. Test on new messages
print("\n" + "=" * 60)
print("TESTING ON NEW M-PESA MESSAGES")
print("=" * 60)

test_messages = [
    "M-PESA confirmed you have received 3000 KES",  # Should be successful
    "Your transaction has been declined",            # Should be failed
    "Payment of 1000 was successful",               # Should be successful
    "Insufficient balance for transaction",         # Should be failed
    "You have received 500 from your friend",       # Should be successful
    "M-PESA failed to process your request",        # Should be failed
]

model.eval()
print("\nPredictions:")
print("-" * 60)

with torch.no_grad():
    for msg in test_messages:
        # Convert to tensor
        x = text_to_tensor(msg, vocab, max_len).unsqueeze(0)
        
        # Predict
        logits, attention = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
        
        # Show result
        status = "SUCCESSFUL" if pred == 1 else "FAILED"
        print(f"   {msg[:40]:<40} → {status} ({confidence:.1%})")

# 7. Show attention (what the model focuses on)
print("\n" + "=" * 60)
print("ATTENTION VISUALIZATION")
print("=" * 60)

sample_msg = "M-PESA confirmed you have received 2000 KES"
x = text_to_tensor(sample_msg, vocab, max_len).unsqueeze(0)

with torch.no_grad():
    _, attn = model(x)
    attn = attn[0].mean(dim=0)  # Average over heads

# Show word importance
words = sample_msg.lower().split()
word_attention = []
for i, word in enumerate(words):
    if i < len(attn):
        word_attention.append((word, attn[i].item()))

print(f"\n Message: {sample_msg}")
print("\n   Word Importance (higher = more influential):")
word_attention.sort(key=lambda x: x[1], reverse=True)
for word, score in word_attention:
    bar = "█" * int(score * 30)
    print(f"   {word:12} {bar:30} {score:.2%}")

print("\n" + "=" * 60)
print("M-PESA Sentiment Analysis Complete!")
print("=" * 60)