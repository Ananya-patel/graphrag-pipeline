# 🕸️ GraphRAG Pipeline

> Project 4 of my RAG Mastery Journey

RAG that understands relationships, not just similarity.
Combines vector search with knowledge graph traversal to answer
multi-hop questions that pure vector RAG cannot.

---

## The Problem This Solves

Vector RAG answers: *"What chunks are similar to my question?"*

GraphRAG answers: *"How are these concepts connected 
through the document?"*
```
Vector only:
"What is Shinto?" → finds Shinto chunk → one fact

GraphRAG:
"How does Shinto connect to Japanese culture?" →
Shinto --[maintains_connection]--> Nature
Shinto --[involves]--> Kami  
Kami --[present_in]--> Nature
Nature --[shapes]--> Japanese Garden Design
→ full relationship chain
```

---

##  Architecture
```
PDF
 ↓
extract_graph.py  → LLM reads each chunk → extracts triples
                    (subject, relation, object)
 ↓
build_graph.py    → triples → NetworkX directed graph
                    + interactive HTML visualization
 ↓
query_graph.py    → extract entities from question
                    → traverse graph (1-hop, 2-hop, path finding)
                    → combine with vector search
 ↓
app.py            → 3-tab UI: Chat + Graph viz + Entity explorer
```

---

##  What Was Built

| Metric | Value |
|---|---|
| Chunks processed | 20 |
| Triples extracted | ~89 |
| Graph nodes (entities) | ~70 |
| Graph edges (relations) | ~85 |
| Retrieval type | Hybrid (vector + graph) |

---

##  Key Concepts Learned

| Concept | What it enables |
|---|---|
| Knowledge graph | Relationship-aware storage |
| Triple extraction | LLM as graph builder |
| Graph traversal | Multi-hop question answering |
| Path finding | "How does X connect to Y?" |
| Hybrid retrieval | Facts + relationships combined |
| Entity disambiguation | Real production challenge |

---

##  GraphRAG vs Vector RAG

| Question type | Vector RAG | GraphRAG |
|---|---|---|
| "What is X?" | ✅ Great | ✅ Great |
| "How does X work?" | ✅ Good | ✅ Good |
| "How did X influence Y?" | ⚠️ Partial | ✅ Full chain |
| "What connects X to Z?" | ❌ Cannot | ✅ Path finding |
| "What is X connected to?" | ❌ Cannot | ✅ Neighbor traversal |

---

##  Setup & Run Locally

**1. Clone and install**
```bash
git clone https://github.com/YOUR_USERNAME/graphrag-pipeline.git
cd graphrag-pipeline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Environment variables**
```
GROQ_API_KEY=your-groq-key-here
```

**3. Add a PDF**
```bash
# Download Japan culture PDF or add your own
python -c "
import requests
headers = {'User-Agent': 'Mozilla/5.0'}
r = requests.get('https://en.wikipedia.org/api/rest_v1/page/pdf/Culture_of_Japan', headers=headers)
open('document.pdf', 'wb').write(r.content)
"
```

**4. Build the graph (one time)**
```bash
python extract_graph.py   # ~30 seconds, extracts triples
python build_graph.py     # builds NetworkX graph + visualization
```

**5. Run the app**
```bash
streamlit run app.py
```

---

##  Project Structure
```
project4/
├── extract_graph.py    # LLM triple extraction from PDF
├── build_graph.py      # NetworkX graph + visualization
├── query_graph.py      # Hybrid retrieval pipeline
├── app.py              # 3-tab Streamlit UI
├── requirements.txt
└── README.md
```

---

##  RAG Mastery Journey

| Project | Topic | Status |
|---|---|---|
| Project 1 | Document Analysis Using LLMs | ✅ Complete |
| Project 2 | RAG System From Scratch | ✅ Complete |
| Project 3 | Multi-Document RAG | ✅ Complete |
| **Project 4** | **GraphRAG Pipeline** | ✅ **Complete** |
| Project 5 | Multi-Doc RAG with LangChain | 🔄 Next |

---
LIVE LINK :
https://graphrag-pipeline.streamlit.app/