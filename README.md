# Doctor RAG

Knowledge Graph-based Retrieval-Augmented Generation (RAG) system for medical literature analysis. Built as part of the master's thesis: **"Mineracao de Texto, Inteligencia Artificial e Aplicacoes em Biotecnologia"**.

The system extracts structured knowledge (entity-relationship triplets) from a medical corpus about **Vitamin D and COVID-19**, stores it in a Neo4j graph database, and provides multiple query strategies to answer medical questions with faithfulness evaluation.

## Architecture

```
corpus.csv (medical literature)
       |
       v
  [utils.py] chunk with overlap
       |
       v
  [RAGout.py] extract triplets via Gemini LLM
       |
       v
  Neo4j Knowledge Graph (nodes + edges + embeddings)
       |
       v
  Query Strategies:
    - Vector-based retrieval
    - Keyword-based search
    - Hybrid (vector + keyword)
    - KG RAG Retriever (semantic + graph navigation)
    - Cypher QA Chain (direct graph queries)
       |
       v
  Faithfulness evaluation + CSV results
```

## Project Structure

```
doctor_rag/
├── RAGout.py              # Main pipeline: builds KG and evaluates query strategies
├── qa_chain.py            # Multi-database Cypher QA (OpenAI gpt-3.5-turbo)
├── qa_index_chain.py      # Multi-strategy evaluation across databases
├── load_data.py           # Loads graph data from CSV into Neo4j
├── utils.py               # Chunking utilities (overlap, whole-document, deduplication)
├── corpus.csv             # Medical literature corpus (sentences)
├── requirements.txt       # Python dependencies
├── environment.yml        # Conda environment
├── .env.example           # Required environment variables
├── server/
│   ├── app.py             # Streamlit chat interface
│   ├── rag.py             # RAG backend (Ollama + Neo4j + LangChain)
│   ├── match.py           # Example Neo4j Cypher graph structure
│   └── graph_neo4j.png    # Knowledge graph visualization
```

## Prerequisites

- Python 3.12
- [Neo4j](https://neo4j.com/) (5.15+)
- [Ollama](https://ollama.ai/) with `mistral` model (for the server component)
- API keys for Google Gemini and/or OpenAI

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/vriez/doctor_rag.git
cd doctor_rag
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate doctor_rag
```

### 2. Start Neo4j

```bash
docker compose up -d
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-secure-password

# For RAGout.py and qa_index_chain.py
GOOGLE_API_KEY=your-google-api-key

# For qa_chain.py and server (if using OpenAI models)
OPENAI_API_KEY=your-openai-api-key

# For multi-database scripts (qa_chain.py, qa_index_chain.py)
# NEO4J_AUTH_MAP='{"db_id": {"username": "neo4j", "password": "...", "url": "bolt://..."}}'
```

## Usage

### Build the Knowledge Graph

```bash
python RAGout.py <PASSWORD> <URL> <DB_ID> <OVERLAP> <EXP_TAG> <CHUNK_SIZE> <MAX_TRIPLETS>
```

Example:

```bash
python RAGout.py mypassword bolt://localhost:7687 mydb 50 experiment1 4096 10
```

This will:
1. Read `corpus.csv` and chunk the text with the specified overlap
2. Extract triplets using Google Gemini
3. Store nodes, edges, and embeddings in Neo4j
4. Evaluate 5 query strategies on 17 multilingual test questions
5. Output results to a CSV with faithfulness scores

### Run the Chat Interface

```bash
cd server
streamlit run app.py
```

Upload a PDF and ask questions through the web interface. The server uses Ollama (mistral) for embeddings and Q&A, with OpenAI (gpt-4) for Cypher query generation.

### Run Multi-Database Evaluation

Set `NEO4J_AUTH_MAP` in your `.env`, then:

```bash
python qa_chain.py          # Cypher QA with OpenAI
python qa_index_chain.py    # Multi-strategy evaluation with Gemini
```

## Models Used

| Component | Model | Provider |
|-----------|-------|----------|
| Triplet extraction & QA | Gemini 1.0 Pro | Google |
| Embeddings | Gemini Embedding-001 | Google |
| Cypher generation | gpt-3.5-turbo / gpt-4 | OpenAI |
| Local LLM & embeddings | Mistral | Ollama |

## Query Strategies

The system evaluates five retrieval approaches:

| Strategy | Description |
|----------|-------------|
| **Vector** | Semantic similarity over embedded graph nodes |
| **Keyword** | Graph keyword matching with tree summarization |
| **Hybrid** | Combined vector + keyword retrieval |
| **KG RAG** | Semantic retrieval with synonym expansion and graph navigation |
| **Cypher Chain** | LLM generates Cypher queries directly against the graph |

Each strategy is tested with varying parameters (`include_text`, `verbose`, `explore_global_knowledge`) and scored using a faithfulness evaluator.
