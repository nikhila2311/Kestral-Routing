# kestrel-routing

Routes Kestrel Home service requests to one of seven teams, to replace the vendor routing bot.
A local text classifier trained on the team that closed each request; unsure requests go to triage with a question to ask.
Includes the pipeline, the test-set predictions, a small web service with one page, and the evaluation evidence.

**No API key is needed.** No paid service is used, and nothing calls the network after `pip install`.

## Deliverables

| File | What it is |
|---|---|
| `predictions.csv` | Predicted team for the 2,178 test requests, in the `sample_submission.csv` format |
| `memo.md` | Memo to the Head of D2C Operations |
| `reports/evaluation.md` | Evaluation evidence (figures in `reports/figures/`) |
| `reports/dashboard.html` | Management dashboard: open reports/dashboard.html in a browser (rebuild: `bash scripts/run_all.sh dashboard`) |

Module contracts are in `docs/contracts.md`, decisions in `PROGRESS.md`.

## Prerequisites

- Python 3.12 or newer (developed and tested on 3.13). Check with `py -0` (Windows) or `python3 --version`.
- Git. On Debian or Ubuntu also the `python3-venv` package.
- Internet access once, for `pip install`.

## Data pack

Copy the eight files of the data pack into `data/raw/` at the repository root (create the folder):

`train.csv`, `resolution_log.csv`, `test_unlabelled.csv`, `sample_submission.csv`, `teams.csv`, `ops-policy.pdf`,
`README.txt`, `email-thread.txt`

They are not in git: policy §10 says customer and operational data must not be published. Trained models are not in git
either, because they can contain customer wording. Both are git-ignored.

## Setup

Run every command in this file from the repository root.

Windows (PowerShell):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run the whole pipeline

macOS / Linux, or Git Bash on Windows:

```bash
bash scripts/run_all.sh clean audit baselines experiments router evaluate train predict
```

Windows PowerShell (plain `bash` there may start WSL, which cannot use the Windows venv):

```powershell
$env:PYTHONHASHSEED = "42"; foreach ($step in "src.clean", "src.audit", "src.baselines", "src.experiments", "src.router", "src.evaluation", "src.predictions train", "src.predictions predict") { .\.venv\Scripts\python -m $step.Split(" "); if ($LASTEXITCODE) { break } }
```

This rewrites the files in `reports/`, builds `models/router.joblib` and writes `predictions.csv`. With the pinned
versions on one machine, `predictions.csv` comes out byte-identical to the committed file.

## Run the service

The service needs the model, built by the **train step**. After the whole pipeline it is already there; otherwise run the
first two commands.

### Windows (PowerShell)

```powershell
.\.venv\Scripts\python -m src.clean
.\.venv\Scripts\python -m src.predictions train
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### macOS / Linux

```bash
.venv/bin/python -m src.clean
.venv/bin/python -m src.predictions train
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

With bash, the three commands are also `bash scripts/run_all.sh clean train serve`.

Then open http://127.0.0.1:8000 in a browser. Stop the service with Ctrl+C.

### Example call

```powershell
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body '{"request_text": "My mixer stopped working"}'
```

```bash
curl -s http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"request_text": "My mixer stopped working"}'
```

| Endpoint | What it does |
|---|---|
| `GET /` | The page |
| `GET /health` | `status`, `model_loaded`, `model_version`, `training_window` |
| `POST /predict` | JSON with `request_text` (1 to 2,000 characters) and optional `channel`, `product_family`, `warranty_status` |

The service has no login and listens on this computer only (127.0.0.1). It does not log or store request text, but answers
quote words from the request, so treat them as customer data.

### If the model is not loaded

The service starts without a model, but `POST /predict` answers 503. The usual causes are that the train step (`src.clean`,
then `src.predictions train`) has not been run, or that scikit-learn, joblib or numpy changed since. Run the train step
again and restart the service.

## Tests

The tests need the data pack, the venv and a finished pipeline run.

| | Windows | macOS / Linux |
|---|---|---|
| Quick (about 2 minutes) | `.\.venv\Scripts\python -m pytest -m "not slow"` | `.venv/bin/python -m pytest -m "not slow"` |
| Full (about 9 minutes) | `.\.venv\Scripts\python -m pytest` | `.venv/bin/python -m pytest` |

The slow tests (marker `slow`) rebuild reports or models from scratch and compare the results.

**Integration test.** Clones the last commit into a temporary folder, copies `data/raw` in, makes a fresh venv, runs the
whole pipeline, checks that `predictions.csv` is byte-identical to the committed file, runs the quick tests and calls the
running service. Prints PASS or FAIL per step, then deletes the folder. It takes about 10 minutes, mostly `pip install` and the pipeline.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\integration_test.ps1
```

```bash
bash scripts/integration_test.sh
```
