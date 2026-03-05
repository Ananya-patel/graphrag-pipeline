import os
import json
import PyPDF2
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
import pickle

load_dotenv()


# ---- PDF Processing ----

def extract_text_from_pdf(pdf_path):
    pages = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append((page_num + 1, text))
    return pages


def split_into_chunks(text, chunk_size=1500, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---- Entity and Relationship Extraction ----

def extract_triples_from_chunk(chunk, client):
    """
    Use LLM to extract entity-relationship triples from a text chunk.
    
    A triple is: subject → relation → object
    Example: "Buddhism" → "originated_in" → "India"
    
    We extract these automatically using the LLM as a parser.
    """

    prompt = f"""You are a knowledge graph extractor.
Read the text below and extract entity-relationship triples.

Rules:
- Extract 3-8 triples per chunk
- Entities should be specific nouns (people, places, religions, 
  art forms, concepts)
- Relations should be simple verb phrases
- Keep entities concise (2-3 words max)
- Only extract what is explicitly stated in the text

Return ONLY a JSON array like this:
[
  {{"subject": "Buddhism", "relation": "originated_in", "object": "India"}},
  {{"subject": "Japan", "relation": "adopted", "object": "Buddhism"}},
  {{"subject": "Shinto", "relation": "focuses_on", "object": "kami spirits"}}
]

No explanation. No markdown. Just the JSON array.

Text:
{chunk}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1  # low temperature = more consistent output
        )

        raw = response.choices[0].message.content.strip()

        # Clean markdown if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        triples = json.loads(raw)

        # Validate structure
        valid = []
        for t in triples:
            if all(k in t for k in ["subject", "relation", "object"]):
                # Clean up entities
                t["subject"] = t["subject"].strip().title()
                t["object"] = t["object"].strip().title()
                t["relation"] = t["relation"].strip().lower()
                valid.append(t)

        return valid

    except Exception as e:
        print(f"    Extraction error: {e}")
        return []


# ---- Process entire document ----

def extract_all_triples(pdf_path, max_chunks=20):
    """
    Process PDF and extract triples from each chunk.
    
    max_chunks: limit chunks to avoid too many API calls.
    20 chunks * ~5 triples each = ~100 triples in the graph.
    That's enough to demonstrate multi-hop traversal.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print("Extracting text from PDF...")
    pages = extract_text_from_pdf(pdf_path)

    # Combine all pages into one text then chunk
    full_text = " ".join(text for _, text in pages)
    chunks = split_into_chunks(full_text)

    # Limit chunks
    chunks = chunks[:max_chunks]
    print(f"Processing {len(chunks)} chunks for triple extraction...")

    all_triples = []

    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}...", end=" ")
        triples = extract_triples_from_chunk(chunk, client)
        all_triples.extend(triples)
        print(f"extracted {len(triples)} triples")

    print(f"\nTotal triples extracted: {len(all_triples)}")
    return all_triples


# ---- Save triples ----

def save_triples(triples, path="triples.pkl"):
    with open(path, "wb") as f:
        pickle.dump(triples, f)
    print(f"Saved {len(triples)} triples to {path}")


def load_triples(path="triples.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---- Preview ----

def preview_triples(triples, n=20):
    print(f"\n=== SAMPLE TRIPLES (first {n}) ===")
    for t in triples[:n]:
        print(f"  {t['subject']} "
              f"--[{t['relation']}]--> "
              f"{t['object']}")

    # Show unique entities
    subjects = set(t["subject"] for t in triples)
    objects = set(t["object"] for t in triples)
    entities = subjects | objects

    print(f"\n=== GRAPH STATS ===")
    print(f"Total triples:    {len(triples)}")
    print(f"Unique entities:  {len(entities)}")
    print(f"Unique relations: "
          f"{len(set(t['relation'] for t in triples))}")

    print(f"\n=== SAMPLE ENTITIES ===")
    print(list(entities)[:20])


if __name__ == "__main__":
    # Check if triples already extracted
    if Path("triples.pkl").exists():
        print("Loading existing triples...")
        triples = load_triples()
    else:
        print("Extracting triples from PDF...")
        triples = extract_all_triples("document.pdf", max_chunks=20)
        save_triples(triples)

    preview_triples(triples)