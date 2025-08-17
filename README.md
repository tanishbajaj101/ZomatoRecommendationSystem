# Restaurant Recommender — Deployment Guide (Streamlit · Qdrant · SQLite)

Production-ready deployment documentation derived from the repository README

## 1. Introduction

An agentic, vector-powered restaurant recommendation app. It uses Sentence Transformers to embed cuisine & ambience, Qdrant for vector search and geo filters, and SQLite for authoritative restaurant data.

Users can like/dislike items (which are then hidden from future recommendations) and review their interaction history.

- Dual-vector semantic search (cuisine + ambience)
- Geo-radius filtering via Qdrant
- Bayesian rating & price context blending with adjustable scoring weights
- Feedback-driven personalization (first-click hard override, momentum updates)
- History UI to review liked/disliked & unhide
- Deterministic user profile persistence (vectors, price prefs, weights, interactions)

## 2. Architecture

Streamlit app (SentenceTransformer, folium/streamlit-folium, SQLite, Qdrant client) connects to Qdrant for named-vector search and geo indexing. The user profile is persisted as vectors and payload in Qdrant.

```
┌──────────────────┐
│ Streamlit (app)  │
│ • SentenceTransformer (MiniLM)
│ • folium / streamlit-folium (map)
│ • SQLite (restaurants.db)
│ • Qdrant client (HTTP)
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│ Qdrant (vectors) │  ← cuisine & ambience named vectors
│ • Geo index      │  ← location:{lat,lon}
│ • User profile   │  ← vectors + price_pref + weights + interactions
└──────────────────┘
```

## 3. Prerequisites

- Python 3.10+ (3.11 recommended)
- SQLite (bundled with Python)
- Qdrant server (Docker or managed/Qdrant Cloud)
- Disk space for model downloads (~100–300MB)

## 4. Environment Variables

Create a .env file (or set these on your platform):

```
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
DB_PATH=restaurants.db
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

> In Docker Compose, QDRANT_URL will be http://qdrant:6333.

## 5. Project Layout (suggested)

```
.
├── app.py
├── restaurants.db
├── README.md
├── requirements.txt
├── .env
├── Dockerfile
└── docker-compose.yml
```

## 6. Quick Start (Local)

```
# 1) Create venv
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -U pip
pip install -r requirements.txt

# 3) Start Qdrant (Docker)
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# 4) Run the app
streamlit run app.py
```

> Open the app at http://localhost:8501
> One-time indexing: In the app sidebar, click “Sync/Refresh Qdrant from SQLite” to vectorize and upsert all restaurants.

## 7. requirements.txt (example)

```
streamlit>=1.33
qdrant-client>=1.7
sentence-transformers>=2.6
numpy>=1.24
folium>=0.15
streamlit-folium>=0.18
```

## 8. Docker (single container)

Dockerfile:

```
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STREAMLIT_SERVER_HEADLESS=true     STREAMLIT_SERVER_ADDRESS=0.0.0.0     STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

> Build: docker build -t resto-app .
> Run: docker run --env-file .env -p 8501:8501 -v $PWD/restaurants.db:/app/restaurants.db resto-app
> Assumes an external Qdrant at QDRANT_URL.

## 9. Docker Compose (app + Qdrant + volumes)

```
version: "3.9"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage

  app:
    build: .
    depends_on:
      - qdrant
    environment:
      QDRANT_URL: "http://qdrant:6333"
      QDRANT_API_KEY: ""
      DB_PATH: "/data/restaurants.db"
      EMBEDDING_MODEL_NAME: "sentence-transformers/all-MiniLM-L6-v2"
      STREAMLIT_SERVER_HEADLESS: "true"
      STREAMLIT_SERVER_ADDRESS: "0.0.0.0"
    ports:
      - "8501:8501"
    volumes:
      - ./restaurants.db:/data/restaurants.db
      - hf_cache:/root/.cache/huggingface

volumes:
  qdrant_storage:
  hf_cache:
```

> Run: docker compose up --build
> Open http://localhost:8501

## 10. Deploying to Cloud (patterns)

Render/Railway/Fly.io/EC2: provision Qdrant (managed or sidecar with persistent volume), deploy the app image, mount restaurants.db as persistent volume, set env vars, expose port 8501.

Google Cloud Run: build & push Docker image, deploy with 1–2 vCPU and 1–2 GB RAM, use managed Qdrant (external), use a mounted volume (e.g., GCS FUSE) if you need a writable restaurants.db.

AWS ECS/Fargate: two containers in a task (qdrant + app) with EFS volumes; ALB to app:8501.

Azure App Service/Container Apps: single container or Compose; Azure Files for persistence; set WEBSITES_PORT=8501.

```
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

> Streamlit Community Cloud requires an external Qdrant endpoint; local SQLite will not persist across cold starts.

## 11. Data & Indexing

SQLite schema (minimum fields):

```
CREATE TABLE IF NOT EXISTS restaurants (
  idx INTEGER PRIMARY KEY,
  Restaurant_Name TEXT,
  Cuisine TEXT,
  Pricing_for_2 INTEGER,
  Dining_Rating REAL,
  Dining_Review_Count INTEGER,
  Website TEXT,
  Address TEXT,
  Latitude REAL,
  Longitude REAL,
  Food_Items TEXT,
  Ambience TEXT,
  Bayesian_Rating REAL,     -- 0..5
  Price_Category INTEGER,   -- 0,1,2,3
  Images TEXT               -- JSON or Python-list-like of URLs
);
```

> Qdrant collections are created automatically:
> • restaurants: named vectors “cuisine” & “ambience” (COSINE), geo index on location
> • user_profiles: vectors + payload (price_pref, has_feedback, score_weights, interactions)
> Use the sidebar Sync/Refresh button after first boot or DB updates.

## 12. Operational Notes

- Model cache: mount a writable volume to /root/.cache/huggingface to avoid re-downloads.
- Security: keep Qdrant private or behind API key; do not expose SQLite; protect Streamlit via proxy (Basic/OIDC).
- Health checks: GET / on the app returns 200; GET http://qdrant:6333/ returns status JSON.

## 13. Using the App

- Change location → map picker → Use this location.
- Adjust Scoring Weights and Price Preference.
- Click Sync/Refresh if you changed the DB.
- Like/Dislike to personalize; items are hidden from future recommendations.
- Manage history in My Interactions (unhide/clear).

## 14. Troubleshooting

- Weights slider only works once: ensure weights are computed before recommendations and sliders have explicit keys.
- No results after interactions: you may have hidden too many items—clear some in My Interactions.
- Model download slow: pre-warm cache or mount HF cache volume in Compose.
- Qdrant connection errors: verify QDRANT_URL, networking, service health; set QDRANT_API_KEY if enabled.
- Images not showing: ensure Images is a JSON array or Python-list-like string of URLs.

## 15. Reproducibility & Upgrades

- Pin requirements.txt for deterministic builds.
- Persist Qdrant storage & SQLite via volumes.
- For schema changes, rebuild the Qdrant index with Sync/Refresh.

## 16. License

MIT (or your organization’s policy).

## 17. Contributing

Open issues for new deployment targets or CI pipelines (e.g., GitHub Actions for Cloud Run/ECS).

PRs welcome for cloud recipes, Helm charts, or Kubernetes manifests.
