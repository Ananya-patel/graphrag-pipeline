import os
import pickle
import faiss
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from build_graph import (
    load_graph, get_neighbors,
    find_path, get_most_connected
)
import PyPDF2

load_dotenv()


# ---- Vector search setup (same as Project 2) ----

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.extract_text()
    return text


def split_into_chunks(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "chunk_id": len(chunks)
            })
        start += chunk_size - overlap
    return chunks


def build_vector_index(chunks, model):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, show_progress_bar=False
    ).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def vector_search(query, model, index, chunks, top_k=3):
    query_emb = model.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "text": chunks[idx]["text"],
            "similarity": round(1 / (1 + dist), 4)
        })
    return results


# ---- Graph query ----

def graph_search(query, G, top_k_entities=3):
    """
    Extract entities from query and traverse graph.
    
    Steps:
    1. Ask LLM what entities are in the question
    2. Find those entities in the graph
    3. Get their neighbors and connections
    4. Return as structured context
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Step 1: Extract entities from question
    entity_prompt = f"""Extract the main entities (people, places, 
religions, art forms, concepts) from this question.
Return ONLY a JSON array of strings.
Example: ["Japan", "Buddhism", "China"]

Question: {query}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": entity_prompt}],
        temperature=0.1
    )

    try:
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        entities = json_parse(raw)
    except Exception:
        entities = []

    print(f"  Entities extracted from question: {entities}")

    # Step 2: Find and traverse each entity
    graph_context = []

    for entity in entities[:top_k_entities]:
        # Get direct neighbors
        outgoing, incoming = get_neighbors(G, entity, depth=1)

        if outgoing or incoming:
            graph_context.append(
                f"\nRelationships for '{entity}':"
            )
            for s, r, o in outgoing[:6]:
                graph_context.append(f"  {s} --[{r}]--> {o}")
            for s, r, o in incoming[:4]:
                graph_context.append(f"  {s} --[{r}]--> {o}")

    # Step 3: Check for paths between entities
    if len(entities) >= 2:
        for i in range(len(entities) - 1):
            path = find_path(G, entities[i], entities[i+1])
            if path:
                graph_context.append(
                    f"\nConnection path "
                    f"{entities[i]} → {entities[i+1]}:"
                )
                for s, r, o in path:
                    graph_context.append(
                        f"  {s} --[{r}]--> {o}"
                    )

    return "\n".join(graph_context), entities


def json_parse(raw):
    """Safe JSON parsing."""
    import json
    raw = raw.strip()
    return json.loads(raw)


# ---- Hybrid RAG: Vector + Graph ----

def hybrid_rag(query, model, vector_index,
               chunks, G, verbose=True):
    """
    Combine vector search and graph traversal.
    
    Vector search  → finds relevant text chunks (facts)
    Graph traversal → finds entity relationships (connections)
    
    Together they give the LLM both facts AND structure.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print(f"\n{'='*60}")
    print(f"Question: {query}")
    print(f"{'='*60}")

    # ---- Vector retrieval ----
    print("\n[1] Vector search...")
    vector_results = vector_search(
        query, model, vector_index, chunks, top_k=3
    )
    for r in vector_results:
        print(f"  [{r['similarity']:.4f}] "
              f"{r['text'][:70]}...")

    # ---- Graph retrieval ----
    print("\n[2] Graph traversal...")
    graph_context, entities = graph_search(query, G)

    if graph_context:
        print(f"  Found graph context for: {entities}")
        print(graph_context[:300] + "...")
    else:
        print("  No graph context found")

    # ---- Combine and generate ----
    print("\n[3] Generating hybrid answer...")

    # Build combined context
    vector_context = "\n\n".join(
        f"[Text chunk {i+1}]\n{r['text']}"
        for i, r in enumerate(vector_results)
    )

    prompt = f"""You are a helpful assistant with access to both 
text passages and a knowledge graph about Japanese culture.

Use BOTH sources to give a comprehensive answer.
The knowledge graph shows explicit relationships between entities.
The text chunks provide detailed factual context.

TEXT CHUNKS:
{vector_context}

KNOWLEDGE GRAPH RELATIONSHIPS:
{graph_context if graph_context else "No graph data found"}

Question: {query}

Give a comprehensive answer using both sources. 
Mention specific relationships from the graph where relevant."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    print(f"\nAnswer:\n{answer}")
    return answer


# ---- Main ----

if __name__ == "__main__":
    import json

    print("Loading components...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    G = load_graph()

    print("Building vector index...")
    text = extract_text_from_pdf("document.pdf")
    chunks = split_into_chunks(text)
    vector_index, _ = build_vector_index(chunks, model)

    print(f"\nGraph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    print(f"Vector index: {len(chunks)} chunks")

    # ---- Test 1: Relationship question ----
    hybrid_rag(
        "How did China influence Japanese culture and writing?",
        model, vector_index, chunks, G
    )

    # ---- Test 2: Multi-hop question ----
    hybrid_rag(
        "What is the connection between Buddhism and Japanese art?",
        model, vector_index, chunks, G
    )

    # ---- Test 3: Compare vector-only vs hybrid ----
    print(f"\n{'='*60}")
    print("COMPARISON: Vector-only vs Hybrid")
    print(f"{'='*60}")

    question = "How are Shinto and nature connected in Japan?"

    print("\n--- VECTOR ONLY ---")
    vector_results = vector_search(
        question, model, vector_index, chunks, top_k=3
    )
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    vector_context = "\n\n".join(
        r["text"] for r in vector_results
    )
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"Answer using only this context:\n\n"
                      f"{vector_context}\n\nQuestion: {question}"
        }]
    )
    print(resp.choices[0].message.content)

    print("\n--- HYBRID (Vector + Graph) ---")
    hybrid_rag(question, model, vector_index, chunks, G)