# Financial AI — Task 1

Equity research pipeline: prices → indicators (no TA-Lib) → news → LLM signal → one-page brief.

Develop locally. Submit as a GitHub repo that reviewers open in Colab.

## Environment: one rule

**Never put API keys in the notebook or in git.**

The same names are used in both places. Only the *storage* changes.

| Where you run | Where keys live | How code reads them |
|---|---|---|
| Your PC | `.env` in this folder | `python-dotenv` |
| Google Colab | Secrets panel (key icon) | `google.colab.userdata` |

`src/config.py` picks the right source automatically.

### Local `.env`

```powershell
copy .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your_key_here
TICKER=AAPL
```

Get a free Groq key at https://console.groq.com/keys

### Colab Secrets

1. Open the notebook in Colab.
2. Click the **key icon** (Secrets).
3. Add `GROQ_API_KEY` (and optionally `TICKER`).
4. Turn **Notebook access** on for each secret.

Do not paste keys into cells. Colab Secrets stay on your Google account, not in GitHub.

## Local run

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install pytest
pytest
```

Open `task1.ipynb` in Cursor / VS Code / Jupyter and run all cells.

### Rule selection

Three momentum rules are compared on holdout data; the winner drives `momentum_bias` via `DEFAULT_RULE_VARIANT` in `src/config.py`.

```powershell
python -m scripts.select_rule
pytest
```

See `docs/RULE_SELECTION.md` for the holdout comparison table and rationale.

## Colab run (after the repo is on GitHub)

Open:

`https://colab.research.google.com/github/lavanblavan/Financial_AI/blob/main/task1.ipynb`

Cell 1 clones this repo into `/content`, installs `requirements.txt`, then imports `src`.

Opening a notebook from GitHub does **not** copy `src/` with it. The clone step is required.

## What is not committed

`.env`, caches, and generated briefs stay local. See `.gitignore`.
