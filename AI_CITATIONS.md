# AI tools used

This project follows the assessment AI policy: AI tools were used with disclosure below. All outputs were reviewed, tested, and interpreted by the candidate.

## Tools

| Tool | Version / access | Purpose |
|------|------------------|---------|
| **Cursor (Composer)** | IDE assistant | Drafting and editing Python modules, tests, notebook markdown, and documentation; debugging and explaining evaluation results |
| **Groq API** | Free tier — `openai/gpt-oss-20b` | Live equity analyst: structured BUY / HOLD / SELL JSON from technical snapshot and headlines |
| **Google Colab** | Free tier | Optional runtime for running the notebook with Secrets-based API keys |

## Where AI was used

### Cursor

- Indicator and evaluation logic (momentum rules, holdout comparison, replay helpers)
- Notebook explanatory text (what each step and chart shows)
- `docs/RULE_SELECTION.md` and README updates
- Reviewing holdout comparison output and selection rationale

### Groq (runtime, not development)

- Generates the analyst signal at notebook run time via `src/signal.py`
- Prompt template lives in `prompts/analyst.md` (written and reviewed by the candidate)
- Model output is validated with `evaluate_llm_output()` (schema, confidence range, consistency checks)

## What was not delegated to AI

- Final choice of holdout evaluation criteria (BUY–SELL spread with minimum sample sizes, then Sharpe)
- Decision to adopt the `score` rule variant based on holdout results documented in `docs/RULE_SELECTION.md`
- Interpretation that results are diagnostic, not proof of trading alpha
- API key handling (`.env` / Colab Secrets — never committed to git)

## Human verification

- `pytest` — 10 tests passing locally
- `python -m scripts.select_rule` — holdout comparison reproduced on AAPL
- Manual review of notebook Run All (local and Colab)
- Confirmation that no secrets appear in the repository

## Cost

All tools listed above are used on **free tiers** only; no paid subscriptions or API spend required to run this project.

---

*Candidate: Lavan · Repository: Financial_AI · Task: Financial AI (Task 1)*
