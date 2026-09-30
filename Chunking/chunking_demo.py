from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)

# Sample Extension Guideline Document
kalro_advisory_doc = """
KALRO EXTENSION BULLETIN: MAIZE LETHAL NECROSIS DISEASE (MLND)

1. DISEASE OVERVIEW
Maize Lethal Necrosis Disease (MLND) is caused by a combined infection of Maize Chlorotic Mottle Virus (MCMV) and Sugarcane Mosaic Virus (SCMV). It can devastate up to 100% of maize yields if not diagnosed early.

2. FIELD IDENTIFICATION SYMPTOMS
- Mild to severe chlorosis (yellowing) starting from leaf bases extending to margins.
- Premature drying of cobs and 'dead-heart' symptoms in young plants.
- Small cobs with poor grain fill.

3. MANAGEMENT & CONTROL STRATEGIES
Cultural Practices:
- Practice strict crop rotation with non-cereal crops such as beans, cowpeas, or Irish potatoes for at least two consecutive seasons.
- Plant certified disease-free seed varieties sourced only from accredited seed dealers.
- Rogue (uproot and burn) infected plants immediately upon detection.

Chemical Control for Insect Vectors:
- Vector insects like thrips and aphids transmit the viruses. Control vectors at early emergence.
- Apply Imidacloprid (200 g/L SL) at a dosage of 15 mL per 20 Litre knapsack sprayer.
- Repeat application after 14 days if vector pressure remains high. Do not spray within 14 days of harvest.
"""

def demonstrate_fixed_chunking(text: str):
    print("==================================================")
    print("1. FIXED-SIZE CHUNKING (Character Count: 250, Overlap: 20)")
    print("==================================================")
    
    # Naive splitter: splits strictly every 250 characters regardless of words or sentences
    fixed_splitter = CharacterTextSplitter(
        separator="",
        chunk_size=250,
        chunk_overlap=20,
    )
    chunks = fixed_splitter.split_text(text)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
        print(repr(chunk))

def demonstrate_semantic_chunking(text: str):
    print("\n==================================================")
    print("2. SEMANTIC / RECURSIVE CHUNKING (Headers & Paragraphs)")
    print("==================================================")
    
    # Structural splitter: Tries section headers ("\n\n"), then newlines ("\n"), preserving paragraphs
    semantic_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "- ", " "],
        chunk_size=400,
        chunk_overlap=50,
    )
    chunks = semantic_splitter.split_text(text)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk.strip())

if __name__ == "__main__":
    demonstrate_fixed_chunking(kalro_advisory_doc)
    demonstrate_semantic_chunking(kalro_advisory_doc)