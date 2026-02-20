🎬 RAG Search Engine

A powerful movie discovery engine that showcases the complete journey through modern information retrieval — starting from classic keyword matching and evolving all the way to intelligent retrieval-augmented generation powered by LLMs.

What started as a collection of CLI experiments has blossomed into a comprehensive search platform with a user-friendly Streamlit interface. The system indexes and searches across 5,000 movies with rich titles and descriptions.

## 🚀 Core Features

- **🔤 Keyword Search** — BM25-powered inverted index for fast, relevance-ranked text retrieval
- **🧠 Semantic Search** — Dense vector embeddings with cosine similarity scoring (full-doc & chunked)
- **⚡ Hybrid Search** — Fusion strategies combining BM25 and semantic search via RRF or weighted normalization
- **✨ Query Enhancement** — Spell checking, query rewriting, and expansion via Gemini
- **🔄 Intelligent Reranking** — LLM-based and cross-encoder reranking to surface top results
- **🤖 RAG (Retrieval-Augmented Generation)** — Retrieve movies then generate answers, summaries, and citations with Gemini
- **📸 Multimodal Search** — Find movies by uploading images using CLIP embeddings
- **📊 Evaluation** — Precision@k, Recall@k, and F1 metrics against a curated golden dataset

## 📋 Prerequisites

- **Python 3.12+** — Modern Python features required
- **uv** — Fast, reliable dependency management
- **Google Gemini API Key** — Required for RAG, query enhancement, reranking, and image features
- **Downloaded Dataset** — See [Download the Dataset](#-download-the-dataset)

---

## 📥 Download the Dataset

The `data/` directory (movies, golden dataset, stopwords) is **not** checked into git. To get the dataset:

- Download from the latest GitHub release:

```bash
git clone https://github.com/<your-username>/rag-search-engine.git
cd rag-search-engine
bash download_data.sh
```

- Or grab it manually from the [Releases page](https://github.com/<your-username>/rag-search-engine/releases).

> The `download_data.sh` script will fetch `data/movies.json`, `data/golden_dataset.json`, and `data/stopwords.txt`.

---

## 🗝️ Google Gemini API Key

To use RAG, query enhancement, reranking, and image features, you need a Google Gemini API key.

- Get your API key here: [Google AI Studio](https://aistudio.google.com/app/apikey)

**Step-by-step:**
1. Go to [Google AI Studio API Key page](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key
5. Add it to your `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

---

## 🔧 Setup & Installation

**Prerequisites:**
- Python 3.12+
- uv (for dependency management & running scripts)
- Google Gemini API key (see above)
- Downloaded dataset (see above)

**Setup:**
1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/<your-username>/rag-search-engine.git
   cd rag-search-engine
   uv sync
   ```
2. Download the dataset:
   ```bash
   bash download_data.sh
   ```
   Or download manually from the [Releases page](https://github.com/<your-username>/rag-search-engine/releases).
3. Add your Gemini API key to `.env`:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

### Initial Setup

Generate the BM25 inverted index (needed for keyword search):

```bash
uv run cli/keyword_search_cli.py build
```

> 💡 **Note:** Embedding caches (`cache/movie_embeddings.npy`, `cache/chunk_embeddings.npy`, etc.) are created automatically on first use.

---

## 🌐 Web Interface (Streamlit)

Launch the interactive web app:

```bash
uv run streamlit run app.py
```

Open http://localhost:8501 — the sidebar gives access to six main sections:

| 📑 Page | ⚙️ Functionality |
|---------|-----------------|
| **🔤 Keyword Search** | BM25 search, keyword queries, and granular scoring tools (TF, IDF, TF-IDF, BM25 TF/IDF) |
| **🧠 Semantic Search** | Vector search (full-doc & chunked), embedding inspection, chunking strategies, model verification |
| **⚡ Hybrid Search** | RRF and weighted hybrid with query enhancement, reranking, debug tracking, LLM evaluation |
| **💬 RAG (Q&A)** | Ask questions, get summaries, citations, or deep answers from retrieved movie documents |
| **📸 Image / Multimodal** | Image-based search (CLIP), or use Gemini to enhance text queries with image context |
| **📊 Evaluation** | Benchmark precision, recall, and F1 against the golden dataset |

---

## 🛠️ CLI Commands

Every feature is also available as a standalone CLI for scripting, debugging, and batch processing.

All commands use `uv run` from the project root.

### 🔤 Keyword Search — `cli/keyword_search_cli.py`

Inverted index, TF-IDF, and BM25 scoring.

```bash
# Initialize the index (do this first)
uv run cli/keyword_search_cli.py build

# Simple keyword search
uv run cli/keyword_search_cli.py search "space adventure"

# Full BM25 ranked search
uv run cli/keyword_search_cli.py bm25search "science fiction aliens" 10

# Low-level scoring inspection
uv run cli/keyword_search_cli.py tf 101 "android"              # Term frequency in doc 101
uv run cli/keyword_search_cli.py idf "android"                # Inverse document frequency
uv run cli/keyword_search_cli.py tfidf 101 "android"          # TF-IDF for doc 101
uv run cli/keyword_search_cli.py bm25tf 101 "android" 1.2 0.7  # BM25 TF (custom k1, b)
uv run cli/keyword_search_cli.py bm25idf "android"            # BM25 IDF for 'android'
```

### 🧠 Semantic Search — `cli/semantic_search_cli.py`

Dense embeddings with sentence-transformers. Full-document or chunked strategies.

```bash
# Verify embedding model loads correctly
uv run cli/semantic_search_cli.py verify

# Embed text and inspect the resulting vector
uv run cli/semantic_search_cli.py embed_text "A lost astronaut searching for home"

# Embed a search query
uv run cli/semantic_search_cli.py embedquery "time travel science fiction"

# Semantic search (full documents)
uv run cli/semantic_search_cli.py search "time travel movies" --limit 5

# Semantic search (chunked for better precision)
uv run cli/semantic_search_cli.py search_chunked "time travel movies" --limit 5

# Experiment with chunking strategies
uv run cli/semantic_search_cli.py chunk "Long text here..." --chunk-size 200 --overlap 20
uv run cli/semantic_search_cli.py semantic_chunk "Long text here..." --max-chunk-size 4

# Build/refresh chunk embeddings for all movies
uv run cli/semantic_search_cli.py embed_chunks

# Validate cached embeddings
uv run cli/semantic_search_cli.py verify_embeddings
```

### ⚡ Hybrid Search — `cli/hybrid_search_cli.py`

Combines BM25 and semantic search using score fusion strategies.

```bash
# Reciprocal Rank Fusion (RRF)
uv run cli/hybrid_search_cli.py rrf-search "funny action movies" --k 60 --limit 5

# Weighted score fusion (alpha: 0–1 controls BM25 vs semantic)
uv run cli/hybrid_search_cli.py weighted-search "dark knight" --alpha 0.5 --limit 5

# Query enhancement before searching
uv run cli/hybrid_search_cli.py rrf-search "scary movie" --enhance rewrite
uv run cli/hybrid_search_cli.py rrf-search "scary movie" --enhance expand
uv run cli/hybrid_search_cli.py rrf-search "scary movie" --enhance spell

# Apply reranking to results
uv run cli/hybrid_search_cli.py rrf-search "scary movie" --rerank-method cross_encoder
uv run cli/hybrid_search_cli.py rrf-search "scary movie" --rerank-method individual
uv run cli/hybrid_search_cli.py rrf-search "scary movie" --rerank-method batch

# Debug: track a specific movie through the pipeline
uv run cli/hybrid_search_cli.py rrf-search "bear movies" --debug "Paddington" --rerank-method cross_encoder

# LLM-as-judge: have Gemini rate relevance of each result
uv run cli/hybrid_search_cli.py rrf-search "bear movies" --evaluate

# Utility: normalize a list of scores
uv run cli/hybrid_search_cli.py normalize 0.5 1.2 3.7 0.1
```

### 💬 RAG & Q&A — `cli/augmented_generation_cli.py`

Retrieve movies and generate natural language responses with Gemini.

```bash
# Answer questions about movies
uv run cli/augmented_generation_cli.py rag "Who directed Inception?"

# Summarize retrieved documents
uv run cli/augmented_generation_cli.py summarize "The Matrix" --limit 5

# Generate answers with source citations
uv run cli/augmented_generation_cli.py citations "Movies with time travel themes" --limit 5

# Detailed comparative analysis
uv run cli/augmented_generation_cli.py question "Compare Interstellar and Gravity" --limit 5
```

### 📸 Multimodal Search — `cli/multimodal_search_cli.py`

CLIP-based image similarity search.

```bash
uv run cli/multimodal_search_cli.py image_search path/to/image.jpg --limit 5
```

### 🖼️ Image-to-Query Rewriting — `cli/describe_image_cli.py`

Send an image + text query to Gemini for intelligent query enhancement.

```bash
uv run cli/describe_image_cli.py --image path/to/poster.jpg --query "find similar movies"
```

### 📊 Evaluation — `cli/evaluation_cli.py`

Benchmark against golden dataset with Precision@k, Recall@k, F1.

```bash
uv run cli/evaluation_cli.py --limit 5
```

---

## 🔀 Pipeline Architecture

```
User Query
    │
    ├─ (Optional) Query Enhancement (Gemini: spellcheck / rewrite / expand)
    │
    ├───────────────┬────────────────────┬───────────────┐
    │               │                    │               │
    ▼               ▼                    ▼               ▼
  BM25         Semantic           Image (CLIP)      (Future: Audio)
 (Inverted     (Embeddings)        Embeddings)
  Index)
    │               │                    │
    └───────┬───────┘                    │
            ▼                            │
      Hybrid Fusion                      │
      (RRF, Weighted, etc.)              │
            │                            │
            ├─ (Optional) Rerank ◄───────┘
            │   (Cross-Encoder / LLM)
            │
            ├─ (Optional) LLM Judge
            │
            ▼
      Top Documents
            │
            ▼
      RAG Generation (Gemini)
      ├─ Q&A
      ├─ Summarization
      ├─ Citations
      └─ Deep Analysis
```

---

## 📂 Project Layout

```
rag-search-engine/
├── app.py                          # Streamlit UI entrypoint
├── pyproject.toml                  # Project dependencies & metadata
├── .env                            # API keys (not tracked)
│
├── cli/                            # CLI tools for each feature
│   ├── keyword_search_cli.py       #   Keyword/BM25 search
│   ├── semantic_search_cli.py      #   Embedding/vector search
│   ├── hybrid_search_cli.py        #   Hybrid fusion (RRF, weighted)
│   ├── augmented_generation_cli.py #   RAG (Q&A, summarization, citations)
│   ├── multimodal_search_cli.py    #   CLIP image search
│   ├── describe_image_cli.py       #   Gemini image-to-query
│   ├── evaluation_cli.py           #   Evaluation/metrics
│   │
│   └── lib/                        # Core logic (shared by CLI & UI)
│       ├── keyword_search.py       #   Inverted index, tokenization, BM25
│       ├── semantic_search.py      #   Embeddings, chunking, similarity
│       ├── hybrid_search.py        #   Score fusion, normalization
│       ├── rag.py                  #   Retrieval + generation
│       ├── llm.py                  #   Gemini API wrappers
│       ├── rerank.py               #   Reranking logic
│       ├── multimodal_search.py    #   CLIP image search
│       ├── evaluation.py           #   Metrics, golden set
│       ├── search_utils.py         #   Data loading, constants
│       │
│       └── prompts/                # Prompt templates (Markdown)
│           ├── answer_question.md
│           ├── answer_question_detailed.md
│           ├── answer_with_citations.md
│           ├── summarization.md
│           ├── spelling.md
│           ├── rewrite.md
│           ├── expand.md
│           ├── individual_rerank.md
│           ├── batch_rerank.md
│           └── llm_judge.md
│
├── data/
│   ├── movies.json                 # Movie dataset (title + description)
│   ├── golden_dataset.json         # Evaluation test cases
│   └── stopwords.txt               # Stopword list
│
└── cache/                          # Generated files (gitignored)
    ├── movie_embeddings.npy        # Full-doc embeddings
    ├── chunk_embeddings.npy        # Chunked embeddings
    ├── chunk_metadata.json         # Chunk-to-movie mapping
    ├── index.pkl                   # BM25 index
    ├── docmap.pkl                  # Doc ID → movie
    ├── term_frequencies.pkl        # Term counts
    └── doc_length.pkl              # Doc lengths
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | 🤖 Google Gemini 2.5 Flash |
| **Sentence Embeddings** | 🧠 all-MiniLM-L6-v2 (sentence-transformers) |
| **Image Embeddings** | 📸 clip-ViT-B-32 (sentence-transformers) |
| **Reranker** | 🔄 cross-encoder/ms-marco-TinyBERT-L2-v2 |
| **Tokenization** | 📖 NLTK Porter Stemmer + custom stopwords |
| **Frontend** | 🌐 Streamlit |
| **Package Manager** | 📦 uv |
| **Language** | 🐍 Python 3.12+ |

---

## 🎯 Key Concepts

### 🔤 BM25 (Keyword Search)
A probabilistic retrieval model that balances term frequency with inverse document frequency, accounting for document length normalization. Perfect for structured queries and terminology matching.

### 🧠 Semantic Search
Converts text to dense vectors and finds similarity based on meaning rather than keywords. Great for conceptual queries and paraphrasing.

### ⚡ Hybrid Fusion
Combines the strengths of both approaches:
- **RRF (Reciprocal Rank Fusion)** — Combines rankings without needing to normalize scores
- **Weighted Normalization** — Blends normalized scores with an alpha parameter (0–1)

### ✨ Query Enhancement
Improves retrieval by rewriting or expanding queries before searching. Strategies include spelling correction, semantic rewriting, and query expansion.

### 🔄 Reranking
Post-retrieval refinement using cross-encoders or LLMs to reorder results by true relevance.

### 🤖 RAG
Retrieves context from documents then generates natural language responses grounded in that context, reducing hallucination and improving accuracy.

---

## 🚦 Quick Start Examples

**I want to search for a movie:**
```bash
uv run cli/keyword_search_cli.py search "space exploration"
# or
uv run cli/hybrid_search_cli.py rrf-search "space exploration"
```

**I want to ask a question about movies:**
```bash
uv run cli/augmented_generation_cli.py rag "What are the best sci-fi movies of all time?"
```

**I want to search by image:**
```bash
uv run cli/multimodal_search_cli.py image_search poster.jpg --limit 5
```

**I want to benchmark search quality:**
```bash
uv run cli/evaluation_cli.py --limit 5
```

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional reranking strategies
- Query expansion via knowledge graphs
- More sophisticated chunking algorithms
- Performance optimizations
- Extended evaluation metrics


---

Happy exploring! 🍿🎬
