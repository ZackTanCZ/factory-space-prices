# ANL488-FYP — Factory Space Price Predictor

ML project for predicting factory space prices ($ per square foot) in Singapore. Built with XGBoost, FastAPI, and Streamlit.

**Live demo**: [Frontend](https://fyp-frontend-latest-3lz9.onrender.com)) | [Backend API](https://fyp-backend-latest-hsqo.onrender.com/docs)

---

## Project Structure

```
├── src/
│   ├── core/           # Pydantic-settings — env config
│   ├── services/       # Inference logic
│   ├── models/         # Pydantic request/response schemas
│   └── pipeline/
│       └── training/   # Training pipeline, MLflow logging, model export
├── backend/            # FastAPI server
├── frontend/           # Streamlit UI
├── config/             # Hydra config files (model, data, API settings)
├── models/             # Champion model artifacts (.pkl)
├── docker/             # Dockerfiles for backend, frontend, MLflow
├── tests/              # Unit tests
└── notebooks/          # Exploratory notebooks (01–09a)
```

---

## Getting Started

### Prerequisites
- Python >= 3.12
- Docker Desktop
- Git

### 1. Clone the repo
```bash
git clone <repo-url>
cd <repo-name>
```

### 2. Create and activate virtual environment
```bash
# Create virtual environment
python -m venv .venv

# Activate — Windows (Git Bash)
source .venv/Scripts/activate

# Activate — Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
# Installs all dependencies (backend + frontend + training) in editable mode
pip install -e ".[all]"
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env` and fill in the values. See `.env.example` for reference.

---

## Running the App

### Option 1 — Local Python (recommended for development)
```bash
# Both backend and frontend together
bash scripts/start.sh

# Or separately
uvicorn backend.api:app --reload --port 8000  # Backend — hot reload on code changes
streamlit run frontend/app.py                  # Frontend
```
- Backend at `http://localhost:8000`
- Frontend at `http://localhost:8501`

### Option 2 — Local Docker (mirrors production)
```bash
docker compose --profile serving up
```
- Backend at `http://localhost:8000`
- Frontend at `http://localhost:8501`

---

## Training a New Model

### 1. Start the MLflow tracking server - Docker Compose
```bash
docker compose --profile training up -d
```
### 1a. Start the MLflow tracking server - Docker build & Docker run
```bash
# Build the image
docker build -f docker/Dockerfile.mlflow -t fyp-mlflow .

# Spin up a container of the image
docker run -p 5000:5000 -v "$(pwd)/mlruns:/mlflow" fyp-mlflow
```
> The `-v` command binds a persistent storage to the container.  
> Experiment and artifacts persist across container restart

> MLflow UI at `http://localhost:5000`

### 2. (Optional) Run hyperparameter optimisation
```bash
# 50 Optuna trials — each trial runs k-fold CV on train set only, test set never touched
python -m src.pipeline.training.hpo --multirun
```
- Search space is configured in `config/hpo_config.yaml`
- Results are written to `logs/optimization_results.yaml` — review best params there
- If the best params look promising, manually update `config/train/model.yaml` before running step 3
- **Note:** HPO optimises CV RMSE on the train set. Always compare test RMSE (from step 3) against the baseline before promoting — HPO does not guarantee a better test score

### 3. Run the training pipeline
```bash
# Trains on full train set, evaluates on held-out test set, logs to MLflow
python -m src.pipeline.training.main
```

### 4. Promote the champion model
- Open the MLflow UI at `http://localhost:5000`
- Compare runs by `rmse` (test set) — lower is better
- Assign the `champion` alias to the best model version in the registry

### 5. Export champion artifacts
```bash
# Downloads champion artifacts from MLflow to models/champion_model/
python -m src.pipeline.training.export_model
```

### 6. Commit and deploy
```bash
git add models/champion_model/
git commit -m "update champion model artifacts"
git push
```
The CI/CD pipeline automatically builds and pushes Docker images to GHCR, then triggers a Render redeploy.

---

## Running Tests
```bash
ruff check .              # Linting
python -m pytest tests/ -v  # Unit tests
```

---

## CI/CD Pipeline

| Trigger | What runs |
|---------|-----------|
| Push to any branch | Lint → Unit tests → Docker build + health checks |
| PR against `main` | Same as above + CD build validation (no push) |
| Merge to `main` | Full CI + build & push images to GHCR + Render redeploy |
| `workflow_dispatch` | Full CI + build & push images to GHCR + Render redeploy |

Branch protection on `main` requires all CI checks to pass before merge.
