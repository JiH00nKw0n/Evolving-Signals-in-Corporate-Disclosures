# Evolving Signals in Corporate Disclosures

Implementation of the **Moving Targets** methodology from Cohen & Nguyen (2024) for measuring evolving signals in corporate earnings call disclosures.

The pipeline extracts performance metrics/targets from earnings call transcripts, calculates Moving Targets (MT) scores measuring how targets change across quarters, and evaluates the predictive power of MT scores on stock returns through Fama-MacBeth regression and calendar-time portfolio backtesting.

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab
```

**Optional** (for Google Gemini LLM extraction):
```bash
pip install google-genai google-auth
```

## Data Preparation

Place the following files in the `data/` folder:

### 1. `data/TRAIN_DATA_FMP_SNP100.csv`

Monthly stock returns data. **Pipe-delimited** (`|`) CSV with columns:

| Column | Description |
|--------|-------------|
| `DATE` | Month-end date (`YYYY-MM-DD`) |
| `SYMBOL` | Ticker symbol |
| `COMPANY_NAME` | Company name |
| `TICKER` | Ticker |
| `ISIN` | ISIN code (must match earnings filenames) |
| `RETURNS` | Current month return |
| `RET_1` | Previous month return |
| `RET_12` | 12-month cumulative return |
| `SIZE` | log(market cap / 1M) |
| `LOG_BM` | log(book equity / market cap) |

### 2. `data/earnings/`

Directory of HTML earnings call transcript files. Filename pattern:

```
{ISIN}={Quarter}_{Year}_{Company_Name}={Date}.html
```

Example: `US0378331005=Q1_2010_Apple_Inc.=2010-04-20.html`

HTML structure: `<strong>` tags for speaker names, `<p>` tags for content.

### 3. Fama-French Factor Data (included)

These files are already included in `data/`:
- `F-F_Research_Data_Factors.tsv` - 3-factor (Mkt-RF, SMB, HML, RF)
- `F-F_Research_Data_5_Factors_2x3.tsv` - 5-factor (+ RMW, CMA)

Tab-delimited, date format `YYYYMM`, values in percentage points.

## Credential Setup

```bash
cp .env.example .env
```

Then edit `.env`:

| Variable | Required For | Description |
|----------|-------------|-------------|
| `OPENAI_API_KEY` | LLM method | OpenAI API key for GPT-5 extraction + text-embedding-3-large |
| `GOOGLE_APPLICATION_CREDENTIALS` | Optional | Path to Google service account JSON for Gemini extraction |
| `FMP_API_KEY` | Optional | FinancialModelingPrep API key (only for `src/fmp.py` data download) |

**Note:** The `base` method (spaCy NER) requires **no API keys** at all.

## Usage

```bash
# Quick single-company test (no API keys needed)
python run.py --mode sample --method base

# Full analysis with all available data
python run.py --mode full --method base

# MT score calculation demo only
python run.py --mode demo

# Use LLM extraction (requires OPENAI_API_KEY)
python run.py --mode full --method llm

# Compare LLM vs NER methods (Ledoit-Wolf 2008 Sharpe ratio test)
python run.py --mode compare --llm-cache output/llm_scores.csv --ner-cache output/ner_scores.csv

# Use cached MT scores (skip recalculation)
python run.py --mode full --cache output/mt_scores_98companies.csv
```

### CLI Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--mode` | `sample`, `full`, `demo`, `compare` | `sample` | Analysis mode |
| `--method` | `base`, `llm` | `base` | Target extraction method |
| `--cache` | path | None | Cached MT score CSV |
| `--llm-cache` | path | None | LLM cache for compare mode |
| `--ner-cache` | path | None | NER cache for compare mode |

## Pipeline Overview

1. **MT Score Calculation** - Extract targets from earnings call HTML transcripts using spaCy NER (`base`) or LLM (`llm`), then compute `MT_t = |Missing Targets at t given Targets at t-4| / |Targets at t-4|`
2. **Fama-MacBeth Regression** - 4 specifications testing MT score as predictor of next-month returns, controlling for size, book-to-market, and return momentum
3. **Portfolio Backtest** - Quintile portfolios with 3-month overlapping holding periods, evaluated with Fama-French 3-factor and 5-factor alphas plus Sharpe ratios (Lo 2002)

## Project Structure

```
├── run.py              # Main entry point (CLI)
├── src/
│   ├── mt_score.py     # Moving Targets score calculation
│   ├── regression.py   # Fama-MacBeth regression analysis
│   ├── backtest.py     # Calendar-time portfolio backtest
│   ├── base_target.py  # spaCy NER target extraction
│   ├── llm_target.py   # LLM-based target extraction
│   ├── embedding.py    # OpenAI embedding + similarity
│   ├── client.py       # API client initialization
│   ├── utils.py        # HTML parsing utilities
│   └── fmp.py          # FinancialModelingPrep API wrapper
├── data/               # Input data (see Data Preparation)
├── output/             # Pipeline output (auto-generated)
├── requirements.txt
└── .env.example
```
