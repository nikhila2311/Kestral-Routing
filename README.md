# kestrel-routing

Routes Kestrel Home service requests to one of seven teams. This file covers running the service; project rules are in
`CLAUDE.md` and module contracts in `docs/contracts.md`.

## Run the service

Needs Python 3.12 or newer. Nothing here uses a paid API, and the service makes no external calls.

The data pack is not in git: copy its files into `data/raw/` first. The model is not in git either, because it can contain
customer wording, so it is built locally by the **train step**. Run every command from the repository root.

### Windows (PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m src.clean
.\.venv\Scripts\python -m src.predictions train
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.clean
.venv/bin/python -m src.predictions train
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

With bash (Git Bash on Windows), the last three commands are also `bash scripts/run_all.sh clean train serve`.

Then open http://127.0.0.1:8000 in a browser. Stop the service with Ctrl+C.

### If the model is not loaded

The service starts without a model, but `POST /predict` answers 503. The usual causes are that the train step (`src.clean`,
then `src.predictions train`) has not been run, or that scikit-learn, joblib or numpy changed since. Run the train step
again and restart the service.

### Endpoints

| Endpoint | What it does |
|---|---|
| `GET /` | The page |
| `GET /health` | `status`, `model_loaded`, `model_version`, `training_window` |
| `POST /predict` | JSON with `request_text` (1 to 2,000 characters) and optional `channel`, `product_family`, `warranty_status` |

```powershell
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body '{"request_text": "My mixer stopped working"}'
```

```bash
curl -s http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"request_text": "My mixer stopped working"}'
```

The service has no login and listens on this computer only (127.0.0.1). It does not log or store request text, but answers
quote words from the request, so treat them as customer data.
