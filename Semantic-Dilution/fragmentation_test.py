import tiktoken

# Load the GPT-4 tokenizer
encoding = tiktoken.get_encoding("cl100k_base")

# ANSI color codes for the terminal (Background colors)
COLOR_1 = "\033[46m\033[30m"  # Cyan background, black text
COLOR_2 = "\033[43m\033[30m"  # Yellow background, black text
RESET = "\033[0m"             # Resets the terminal back to normal

# Our mini dataset
sentences = {
    "English": "High quality leather shoes for men",
    "Swahili": "Viatu vya ngozi vya ubora wa juu kwa wanaume",
    "Sheng": "Fegi za ngware high quality za maboys"
}

for language, text in sentences.items():
    # 1. Encode text and get counts
    token_ids = encoding.encode(text)
    word_count = len(text.split())
    ratio = len(token_ids) / word_count
    
    # 2. Print the stats
    print(f"\n--- {language} ---")
    print(f"Words: {word_count} | Tokens: {len(token_ids)} | Ratio: {ratio:.2f}")
    
    # 3. Print the color-coded fragments
    print("Fragmentation: ", end="")
    for i, token_id in enumerate(token_ids):
        # Decode the single token ID back into a text chunk
        chunk = encoding.decode([token_id])
        
        # Alternate colors: if 'i' is even, use COLOR_1, else use COLOR_2
        current_color = COLOR_1 if i % 2 == 0 else COLOR_2
        
        # Print the colored chunk (end="" keeps it all on one line)
        print(f"{current_color}{chunk}{RESET}", end="")
        
    print() # Adds a blank line before the next language