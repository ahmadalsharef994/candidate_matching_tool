# Candidate Matching Tool

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/NLP-sentence--transformers-orange" alt="NLP">
  <img src="https://img.shields.io/badge/FastAPI-REST%20API-green?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/cosine--similarity-matching-purple" alt="Matching">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</p>

An **AI-powered candidate-to-job matching tool** that uses sentence transformers and cosine similarity to rank candidates against job descriptions. Removes keyword-matching bias and understands semantic relevance.

---

## 🏗️ Matching Pipeline

```mermaid
flowchart LR
    JD["📋 Job Description"] --> Encoder["🤖 Sentence\nTransformer\n(all-MiniLM-L6-v2)"]
    Resume["📄 Candidate\nResume"] --> Encoder
    Encoder --> JD_Embed["JD Embedding\n(384-dim)"]
    Encoder --> CV_Embed["CV Embedding\n(384-dim)"]
    JD_Embed --> Cosine["📐 Cosine\nSimilarity"]
    CV_Embed --> Cosine
    Cosine --> Score["🏆 Match Score\n(0 → 1)"]
    Score --> Ranked["📊 Ranked\nCandidates"]
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

### Match via API

```bash
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Senior Python backend engineer...",
    "candidates": [
      {"id": "c1", "resume": "5 years Python, FastAPI, PostgreSQL..."},
      {"id": "c2", "resume": "React developer with 3 years experience..."}
    ]
  }'
```

**Response:**
```json
{
  "ranked": [
    {"id": "c1", "score": 0.89, "rank": 1},
    {"id": "c2", "score": 0.41, "rank": 2}
  ]
}
```

---

## 📊 Features

- Semantic matching (not just keyword overlap)
- Batch scoring of multiple candidates
- Configurable similarity threshold
- CSV bulk import/export

---

## 📄 License

MIT
