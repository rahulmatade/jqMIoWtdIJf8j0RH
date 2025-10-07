# README — PotentialTalents.ipynb

## Overview

This notebook implements a  machine‑learning ranking pipeline that aims to rank candidate profiles for a given job description. The approach combines:

* text embeddings (Sentence‑Transformer models) and cosine similarity as baseline signals,
* a LightGBM ranker trained to produce a model ranking (fit score) that can be compared with the baseline similarity ranking.

## Dataset & features 

* **Dataset file:** `/content/potential-talents.csv`

* **Primary columns referenced in the notebook:**

  * `id` — unique identifier for each candidate
  * `job_title` — candidate text field used as the textual profile
  * `connection` — a numeric / string field that is cleaned and normalized

* **Derived / computed columns:**

  * `_text` — cleaned text column created from `job_title` (used for encoding)
  * `sim_minilm`, `sim_mpnet`, `sim_distil` — cosine similarity scores between candidate embeddings and the job description embedding using the following Sentence‑Transformer models:

    * `all-MiniLM-L6-v2` (MiniLM)
    * `sentence-transformers/all-mpnet-base-v2` (MPNet)
    * `sentence-transformers/multi-qa-distilbert-cos-v1` (DistilBERT QA model)
    * `conn_norm` — normalized version of `connection` (using a `StandardScaler` inside function `normalize_connections_array`)

* **Target / label used for training:**

  * `starred_label` — a binary target created from a hard‑coded list `STARRED_IDS` (where `y = 1` for ids in `STARRED_IDS`, else 0). This is used as the supervision signal for the ranker.

## Model choice

* **Model used:** `lightgbm.LGBMRanker` (LightGBM Ranker)

* **Key configurations:**

  * `objective='lambdarank'`
  * `metric='ndcg'`
  * `n_estimators=200`
  * `importance_type='gain'`

* **Baseline for comparison:** a cosine‑similarity baseline using MiniLM embeddings (`sim_scores`) is computed and min‑max normalized. The pipeline presents both the model ranking (`fit_score_model`) and the baseline similarity ranking (`fit_score_similarity`) for comparison.

---

## Outcome

The pipeline produces / displays the following outputs :
1. **Baseline similarity ranking** (`fit_score_similarity`) computed from cosine similarity (MiniLM) and min‑max normalized.
2. **Trained LGBMRanker**: the ranker is trained on the assembled features.
3. **`df_result`** — a DataFrame that contains the original records plus new columns:

   * `fit_score_model` — model predicted score
   * `fit_score_similarity` — baseline similarity score
   * `starred_label` — binary label used as supervision
   * `sim_minilm`, `sim_mpnet`, `sim_distil` — similarity features
4. **Top‑K displays**: the notebook shows two top‑K candidate lists for inspection:

   * Top‑K by baseline similarity
   * Top‑K by the model (`fit_score_model`)
5. **Feature importances chart** for the trained ranker is plotted.
