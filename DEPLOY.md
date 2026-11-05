# Cloud & Local Deployment Guide — Route.it (SIH 2026 PS SIH26137)

The repository is configured for **Unified Single-Service Hosting**: FastAPI (`api/main.py`) now serves both the REST/WebSocket backend (`/api/*`, `/ws/*`, `/docs`) **and** the prebuilt React + MapLibre production frontend (`dashboard/dist`) on a single port (`8000` locally or `$PORT` in the cloud).

---

## Option 1: 1-Click Free Cloud Deployment on Render (Recommended)
1. Push this repository folder to a GitHub repository.
2. Go to [Render Dashboard](https://dashboard.render.com/) -> **New +** -> **Blueprint** (or **Web Service**).
3. Connect your GitHub repository. Render will automatically detect `render.yaml`:
   - **Build Command**: `pip install -e .`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Click **Apply / Deploy**. Within ~2 minutes, your live public URL (`https://route-it-sih2026.onrender.com`) will serve the full interactive dashboard, solver API, and WebSockets on one link!

---

## Option 2: 1-Click Cloud Deployment on Railway or Fly.io (Docker)
1. Push this repository to GitHub.
2. On [Railway](https://railway.app/), click **New Project** -> **Deploy from GitHub repo**.
3. Railway automatically reads `railway.json` and `Dockerfile`, builds the container, and gives you a public `https://...up.railway.app` domain with `/health` monitoring enabled.

---

## Option 3: Local Single-Command Production Mode
Because `dashboard/dist` is prebuilt and mounted directly into FastAPI, you don't even need two terminals to run the full stack locally:
```bash
pip install -e .
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your browser — both the React dashboard and FastAPI backend will run together on port `8000`!
