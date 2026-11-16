# PDF Ingestion and Semantic Search with LangChain and Postgres

A command-line **RAG** (*Retrieval-Augmented Generation*) system: it reads a PDF, stores its content as vectors in PostgreSQL + pgVector, and answers user questions using **only** what is in the document.

## Table of contents

- [About the project](#about-the-project)
- [How it works](#how-it-works)
- [How to run](#how-to-run)
- [Example questions](#example-questions)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## About the project

The project has two stages:

1. **Ingestion:** `document.pdf` is loaded and split into 1000-character *chunks* (150-character overlap). Each chunk is turned into an *embedding* (model set in `.env`) and the vectors are stored in PostgreSQL with the pgVector extension.
2. **Search (chat):** the user asks a question in the terminal. The question is embedded, the 10 closest chunks are fetched from the database and sent to an LLM together with a prompt that forces it to answer only from that context. If the information is not in the context, the answer is:

   > Não tenho informações necessárias para responder sua pergunta.
   > (*"I don't have the information needed to answer your question."*)

The `document.pdf` in this repository is a table of fictional companies with three columns: **company name**, **revenue** and **founding year** (34 pages, about 980 rows). The prompt and the answers are in Portuguese.

**Stack:** Python 3.13, LangChain, PostgreSQL 17 + pgVector (via Docker Compose), and the Google provider for embeddings and for the LLM (see [Model provider](#model-provider)).

## How it works

```
INGESTION
document.pdf ─► PyPDFLoader ─► RecursiveCharacterTextSplitter ─► Embeddings ─► PGVector

SEARCH
question ─► similarity_search_with_score(k=10) ─► prompt (CONTEXT + RULES + QUESTION) ─► LLM ─► answer
              └─ embeds the question with the same model used at ingestion, then finds the closest chunks in pgVector
```

`PGVector` is created with the embedding model, so both `add_documents` (ingestion) and `similarity_search_with_score` (search) call it internally. The question and the chunks must always be embedded by the same model.

## How to run

### Prerequisites

- **Python 3.13** (pinned in `mise.toml`: 3.13.18). Two options:
  - With [mise](https://mise.jdx.dev/): from the project root, run `mise install` and it installs the right version.
  - Without mise: install Python 3.13 yourself (any 3.13.x). Other versions were not tested.
- [Docker](https://www.docker.com/) and Docker Compose
- A Google API key (create one in [Google AI Studio](https://aistudio.google.com/apikey))

### 1. Install the dependencies

Check the Python version:

```bash
python --version   # Python 3.13.x
```

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure the environment variables

```bash
cp .env.example .env
```

Fill in the `.env`:

| Variable | Description | Example |
| --- | --- | --- |
| `API_KEY` | API key of the model provider (Google) | `<your-key>` |
| `LLM_MODEL` | LLM, in the `provider:model` format | `google_genai:<llm-model>` |
| `EMBEDDING_MODEL` | Embedding model | `<embedding-model>` |
| `PGVECTOR_URL` | Database connection string | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PGVECTOR_COLLECTION` | Name of the vector collection | `my_collection` |

Model names are up to you: any LLM and embedding model offered by the provider works. Model availability changes often, so check the provider's official documentation for the valid names, and prefer light, cheap models, which are enough for this data volume.

#### Model provider

The project was tested with Google's **Gemini**, for both embeddings and the LLM, and the code is wired to the **Google** provider:

- `src/vector_store.py` creates the embeddings with `GoogleGenerativeAIEmbeddings`.
- `src/search.py` creates the LLM with LangChain's `init_chat_model`, using the `provider:model` format set in `LLM_MODEL`.

To use another provider, you need to change the code:

1. In `src/vector_store.py`, replace `GoogleGenerativeAIEmbeddings` with the embeddings class of your provider (and install its LangChain package).
2. Set `LLM_MODEL` to the matching `provider:model` value and use a key from the same provider in `API_KEY`.

### 3. Start the database

```bash
docker compose up -d
```

This starts PostgreSQL with pgVector on port `5432` and creates the `vector` extension.

### 4. Ingest the PDF

```bash
python src/main.py ingest document.pdf
```

The argument is the path to the PDF, relative to the project root.

> **Rate limits:** many free plans limit embedding requests per minute (each chunk counts as one). For that reason, ingestion sends the chunks in batches of 90 and waits 61 seconds between batches. When there are more than 90 chunks, it takes a few extra minutes. Adjust `EMBED_BATCH_SIZE` and `EMBED_BATCH_INTERVAL_SECONDS` in `src/ingest.py` if your plan has a different limit.

### 5. Open the chat

```bash
python src/main.py chat
```

Type your question and press Enter. To quit, type `sair`, or use `Ctrl+C` / `Ctrl+D`.

### Stop and clean up

```bash
docker compose stop        # stops the containers and keeps the data
docker compose start       # starts them again
docker compose down -v     # removes containers, network and volumes (deletes the vectors)
```

If you change the embedding model, delete the data with `docker compose down -v` and ingest again. Different models produce vectors with different dimensions, and ingestion fails against the old dimension.

## Example questions

The examples below were run against the `document.pdf` in this repository. The questions and answers are in Portuguese because the prompt is in Portuguese.

### Questions the system answers

These are questions about **one specific company**: the name appears in the question and its table row is in the retrieved chunks.

```
Pergunta: Qual foi o faturamento da Alfa Saúde LTDA em 1996?
RESPOSTA: O faturamento da Alfa Saúde LTDA é de R$ 131.151.997,92.
```

```
Pergunta: Qual o faturamento da empresa Alfa Agronegócio Indústria e em que ano ela foi fundada?
RESPOSTA: O faturamento da empresa Alfa Agronegócio Indústria é de R$ 85.675.568,77 e ela foi fundada em 1931.
```

### Out-of-context questions

The LLM follows the rule and declines. General-knowledge questions, data the PDF does not have, and companies that are not in it:

```
Pergunta: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

```
Pergunta: Qual o faturamento da empresa Inexistente Tecnologia S.A.?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Project structure

```
├── docker-compose.yml     # PostgreSQL + pgVector
├── requirements.txt       # Dependencies
├── .env.example           # Environment variables template
├── mise.toml              # Python version
├── document.pdf           # PDF used for ingestion
└── src/
    ├── main.py            # Entry point: `ingest` and `chat` commands
    ├── config.py          # Reads and validates the .env
    ├── vector_store.py    # PGVector connection and embedding model
    ├── ingest.py          # Ingestion: loading, chunking and storing the vectors
    ├── search.py          # Similarity search, prompt and LLM call
    └── chat.py            # Terminal chat loop
```

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'langchain_core'` | The virtual environment is not active or the dependencies were not installed. Run `source .venv/bin/activate` and `pip install -r requirements.txt`. |
| `429 Quota exceeded` during ingestion | Request limit of your provider plan. Wait a minute and run the ingestion again, or lower `EMBED_BATCH_SIZE` in `src/ingest.py`. |
| Credentials error when opening the chat | `API_KEY` is empty or missing in the `.env`. |
| Database connection error | The containers are not running. Run `docker compose up -d` and check with `docker compose ps`. |
| Vector dimension error during ingestion | The embedding model changed after the first ingestion. Run `docker compose down -v` and ingest again. |
