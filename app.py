import os
import pickle
import json
import faiss
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pathlib import Path
import PyPDF2
from build_graph import (
    load_graph, get_neighbors,
    find_path, get_most_connected,
    visualize_graph
)

load_dotenv()


# ---- Helper functions ----

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
    return index


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


def extract_entities(query):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"""Extract main entities from this question.
Return ONLY a JSON array of strings.
Example: ["Japan", "Buddhism"]
Question: {query}"""
        }],
        temperature=0.1
    )
    try:
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return []


def graph_search(query, G):
    entities = extract_entities(query)
    graph_context = []

    for entity in entities[:3]:
        outgoing, incoming = get_neighbors(G, entity, depth=1)
        if outgoing or incoming:
            graph_context.append(
                f"\nRelationships for '{entity}':"
            )
            for s, r, o in outgoing[:6]:
                graph_context.append(f"  {s} --[{r}]--> {o}")
            for s, r, o in incoming[:4]:
                graph_context.append(f"  {s} --[{r}]--> {o}")

    if len(entities) >= 2:
        path = find_path(G, entities[0], entities[-1])
        if path:
            graph_context.append(
                f"\nConnection path "
                f"{entities[0]} → {entities[-1]}:"
            )
            for s, r, o in path:
                graph_context.append(
                    f"  {s} --[{r}]--> {o}"
                )

    return "\n".join(graph_context), entities


def hybrid_answer(query, model, index, chunks, G):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    vector_results = vector_search(
        query, model, index, chunks, top_k=3
    )
    graph_context, entities = graph_search(query, G)

    vector_context = "\n\n".join(
        f"[Text chunk {i+1}]\n{r['text']}"
        for i, r in enumerate(vector_results)
    )

    prompt = f"""You are a helpful assistant with access to 
text passages and a knowledge graph.

Use BOTH to give a comprehensive answer.
Cite graph relationships explicitly where relevant.
If the answer isn't in the context say so.

TEXT CHUNKS:
{vector_context}

KNOWLEDGE GRAPH:
{graph_context if graph_context else "No graph data found"}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return (
        response.choices[0].message.content,
        vector_results,
        graph_context,
        entities
    )


# ---- Streamlit UI ----

st.set_page_config(
    page_title="GraphRAG",
    page_icon="🕸️",
    layout="wide"
)

st.title("🕸️ GraphRAG — Knowledge Graph + Vector Search")
st.caption("Project 4 — Relationship-aware retrieval")


@st.cache_resource
def load_components():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    G = load_graph()
    text = extract_text_from_pdf("document.pdf")
    chunks = split_into_chunks(text)
    index = build_vector_index(chunks, model)
    return model, G, chunks, index


model, G, chunks, index = load_components()

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs([
    "💬 Ask Questions",
    "🕸️ Knowledge Graph",
    "🔍 Explore Entities"
])


# ---- Tab 1: Chat ----
with tab1:
    st.subheader("Hybrid RAG — Vector + Graph")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("graph_context"):
                    with st.expander("🕸️ Graph relationships used"):
                        st.code(msg["graph_context"])
                if msg.get("vector_results"):
                    with st.expander("📄 Text chunks used"):
                        for r in msg["vector_results"]:
                            st.caption(
                                f"Similarity: {r['similarity']:.4f}"
                            )
                            st.caption(r["text"][:200] + "...")
                            st.divider()

    query = st.chat_input(
        "Ask about relationships, connections, influences..."
    )

    if query:
        with st.chat_message("user"):
            st.write(query)
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("assistant"):
            with st.spinner(
                "Searching vectors + traversing graph..."
            ):
                answer, v_results, g_context, entities = \
                    hybrid_answer(query, model, index, chunks, G)

            st.write(answer)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Entities found", len(entities))
            with col2:
                st.metric(
                    "Graph relationships",
                    len(g_context.split("\n")) if g_context else 0
                )

            if g_context:
                with st.expander("🕸️ Graph relationships used"):
                    st.code(g_context)

            with st.expander("📄 Text chunks used"):
                for r in v_results:
                    st.caption(
                        f"Similarity: {r['similarity']:.4f}"
                    )
                    st.caption(r["text"][:200] + "...")
                    st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "graph_context": g_context,
            "vector_results": v_results
        })


# ---- Tab 2: Graph Visualization ----
with tab2:
    st.subheader("Interactive Knowledge Graph")
    st.caption(
        f"{G.number_of_nodes()} entities · "
        f"{G.number_of_edges()} relationships"
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        max_nodes = st.slider("Max nodes to show", 20, 80, 40)
        if st.button("🔄 Regenerate Graph"):
            visualize_graph(G, "graph.html", max_nodes)
            st.rerun()

    # Generate if not exists
    if not Path("graph.html").exists():
        visualize_graph(G, "graph.html", max_nodes)

    with open("graph.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    components.html(html_content, height=600, scrolling=True)

    # Most connected entities
    with col1:
        st.subheader("🏆 Most Connected")
        for node, degree in get_most_connected(G, top_n=10):
            st.markdown(f"**{node}** — {degree}")


# ---- Tab 3: Entity Explorer ----
with tab3:
    st.subheader("Explore Entity Relationships")

    all_entities = sorted(list(G.nodes()))
    selected = st.selectbox(
        "Select an entity:", all_entities
    )

    depth = st.radio("Search depth:", [1, 2], horizontal=True)

    if selected:
        results, _ = get_neighbors(G, selected, depth=depth)

        if results:
            st.markdown(
                f"**{len(results)} relationships** "
                f"found for '{selected}':"
            )
            for s, r, o in results:
                col1, col2, col3 = st.columns([2, 2, 2])
                with col1:
                    st.info(s)
                with col2:
                    st.caption(f"──[{r}]──▶")
                with col3:
                    st.success(o)
        else:
            st.warning(f"No relationships found for '{selected}'")

        # Path finder
        st.divider()
        st.subheader("🗺️ Find Connection Path")
        target = st.selectbox(
            "Connect to:",
            [e for e in all_entities if e != selected]
        )

        if st.button("Find Path"):
            path = find_path(G, selected, target)
            if path:
                st.success(
                    f"Found path in {len(path)} hops!"
                )
                for s, r, o in path:
                    st.markdown(
                        f"**{s}** ──[{r}]──▶ **{o}**"
                    )
            else:
                st.warning("No path found between these entities")