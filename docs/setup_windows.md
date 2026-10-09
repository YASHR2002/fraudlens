# Windows Setup

Everything below runs in **PowerShell** on Windows 11. Tested on an 8 GB RAM laptop with
Docker Desktop on WSL2.

## 1. Cap Docker's memory (WSL2)

Docker Desktop runs inside a WSL2 VM that by default can take up to half your RAM and does not
give it back promptly. On an 8 GB machine that starves VS Code and model training, so cap it.

Create `C:\Users\<your-name>\.wslconfig` (note the leading dot, no file extension):

```powershell
@"
[wsl2]
memory=3GB
processors=2
swap=4GB
"@ | Set-Content -Encoding ascii "$env:USERPROFILE\.wslconfig"
```

Then restart WSL and Docker so the limits apply:

```powershell
# Quit Docker Desktop first (system tray icon > Quit Docker Desktop), then:
wsl --shutdown
# Start Docker Desktop again from the Start menu, wait until it says "Engine running", then check:
docker info --format "{{.MemTotal}}"   # should be about 3 GB (printed in bytes, ~3.1e9)
```

> The project's Docker Compose file uses **profiles** so that only the containers a phase
> needs are running (see the README). Every container also has its own memory limit.

## 2. Install `uv` and create the environment

[`uv`](https://docs.astral.sh/uv/) manages Python versions, the virtual environment, and
dependencies (pinned in `uv.lock`).

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Open a new terminal so `uv` is on PATH, then from the project folder:
uv --version
uv sync                 # downloads Python 3.11 if needed, creates .venv, installs all groups
uv run python -m fraudlens --help
```

`uv run <cmd>` runs a command inside `.venv` without activating it. To activate it instead:

```powershell
.\.venv\Scripts\Activate.ps1
python -m fraudlens --help
```

If activation is blocked, allow local scripts once for your user:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Dependencies are split into groups. `uv sync` installs all of them; you can install a subset:

| Group | Contents | Install only this |
|---|---|---|
| core | pandas, pyarrow, scikit-learn, XGBoost, LightGBM, SHAP, google-genai, Typer | `uv sync --no-default-groups` |
| `train` | MLflow, Optuna, SQLAlchemy/psycopg, Kaggle, Evidently, matplotlib | `--group train` |
| `api` | FastAPI, Uvicorn, Prometheus instrumentator, Hugging Face Hub | `--group api` |
| `dashboard` | Streamlit, Plotly, httpx | `--group dashboard` |
| `dev` | pytest, ruff, pre-commit | `--group dev` |

**Notebooks in VS Code:** open the notebook, click **Select Kernel** (top right) >
**Python Environments** > **.venv (Python 3.11)**. Another kernel (e.g. a system Python) fails
with `ModuleNotFoundError: No module named 'fraudlens'`, because the package is installed only
in `.venv`.

Install the Git hooks once (runs ruff lint and format on each commit):

```powershell
uv run pre-commit install
uv run pre-commit run --all-files
```

**Fallback without uv:** `py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -e .`
then install the groups you need, e.g. `pip install --group dev` (pip 25.1+), or export them
with `uv export --group dev --format requirements-txt`.

## 3. Environment file and Gemini API key

```powershell
Copy-Item .env.example .env
```

Get a free Gemini API key:

1. Go to [Google AI Studio](https://aistudio.google.com/apikey) and sign in with a Google account.
2. Click **Create API key** and copy it.
3. Paste it into `.env` as `GOOGLE_API_KEY=...`.
4. `GEMINI_MODEL` is preset to `gemini-3.8-flash` (the newest stable Flash model on the free
   tier as of October 2026). Model names change; see the
   [Gemini models list](https://ai.google.dev/gemini-api/docs/models).

Also change `POSTGRES_PASSWORD` in `.env` **before the first** `docker compose up`: PostgreSQL
sets the password only when its data volume is created (afterwards: `docker compose --profile "*" down -v`
deletes the volume, then start again and re-run `fraudlens load-db`). `.env` is git-ignored; never commit it.

Check that the configuration loads (secrets are printed masked):

```powershell
uv run python -m fraudlens show-config
```

## 4. Get the dataset

The dataset is the Sparkov
[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
on Kaggle (CC0 licence, about 500 MB unzipped).

**Option A, manual (simplest):** sign in to Kaggle, open the page, click **Download**, and
extract `fraudTrain.csv` and `fraudTest.csv` into `data
aw\`.

**Option B, Kaggle API.** The `kaggle` CLI (version 2.x, installed by `uv sync`) no longer
uses `kaggle.json`. Authenticate once with either:

```powershell
uv run kaggle auth login                     # browser sign-in; credentials are cached
# or: create a token at https://www.kaggle.com/settings/api ("Generate New Token") and
$env:KAGGLE_API_TOKEN = "<token>"            # current session only
```

Then:

```powershell
uv run python -m fraudlens download-data     # skips the download if the files are already there
```

## 5. Start PostgreSQL

Start **Docker Desktop** first, then:

```powershell
docker compose --profile data up -d
docker compose ps                      # postgres should be "healthy"
docker compose --profile "*" down       # stop when done
```

PostgreSQL listens only on `127.0.0.1:5432`. Use `127.0.0.1`, not `localhost`, in `.env`:
on Windows `localhost` is tried as IPv6 `::1` first, which Docker Desktop leaves hanging for
about two minutes before falling back. Its data lives in the `fraudlens_pgdata` volume;
`docker compose --profile "*" down -v` deletes it. Always pass `--profile "*"` to `down`: a plain
`docker compose down` ignores every service that belongs to a profile, which here is all of them.
