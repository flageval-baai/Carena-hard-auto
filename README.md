
## Carena-Hard-Auto: Automatic Benchmark Construction and Bayesian Ranking

This repository contains the **full evaluation pipeline** used in our paper, including:

1. automatic answer generation from multiple LLMs,
2. LLM-as-a-Judge based pairwise evaluation, and
3. Bayesian Bradley–Terry (Bayes-BT) ranking with robustness analysis.

The benchmark prompts are automatically curated using a modified BenchBuilder pipeline
(see `BenchBuilder(modified)/`), and the final evaluation follows a **model-agnostic,
noise-aware ranking framework**.

---

## Repository Structure

```text
.
├── BenchBuilder(modified)/     # Prompt curation pipeline (see separate README)
├── Carena_hard_auto.json       # Final benchmark prompts
├── gen_answer.py               # Generate model answers
├── model_answers/              # Saved model outputs
├── Bayes_bt.py                 # Bayesian BT ranking with LLM judge
├── noise_analysis.py           # Noise & robustness analysis
└── topic_model_dir/            # Topic model / clustering artifacts
````

---

## Overview of the Pipeline

The evaluation pipeline consists of **three main stages**:

1. **Answer Generation**
   Each candidate model generates answers for the same benchmark prompts.

2. **LLM-as-a-Judge Evaluation + Bayesian Ranking**
   A judge model performs pairwise comparisons between model outputs, and rankings are inferred using a Bayesian Bradley–Terry model.

3. **Noise & Robustness Analysis**
   We analyze the stability of rankings under different noise conditions and judge variability.

Each stage is implemented as a standalone script, enabling easy reproduction and extension.

---

## 1. Benchmark Prompts

The benchmark prompts are stored in:

```text
Carena_hard_auto.json
```

This file contains the final set of high-quality prompts automatically curated from
crowdsourced data using our modified BenchBuilder pipeline.

Each entry includes:

* the prompt text,
* topic / cluster metadata,
* unique identifiers used throughout the evaluation.

---

## 2. Generating Model Answers

### Script: `gen_answer.py`

This script queries each evaluated model to generate responses for all benchmark prompts.

#### Usage

```bash
python gen_answer.py \
  --model MODEL_NAME \
  --input Carena_hard_auto.json \
  --output model_answers/MODEL_NAME.jsonl
```

* `MODEL_NAME` can be any API-based or local LLM supported by your inference backend.
* Generated answers are saved in `model_answers/` in JSONL format.

Each output entry contains:

* prompt ID,
* prompt text,
* model name,
* generated answer.

> **Note**
> All models are evaluated on the *exact same prompt set* to ensure fair comparison.

---

## 3. LLM-as-a-Judge Evaluation & Bayesian Ranking

### Script: `Bayes_bt.py`

This script implements our **Bayesian Bradley–Terry (Bayes-BT) ranking framework**.

#### Key Ideas

* A judge LLM performs **pairwise comparisons** between answers from different models.
* Each comparison yields a preference signal.
* A Bayesian BT model aggregates noisy judgments into:

  * posterior skill distributions,
  * a global model ranking with uncertainty estimates.

This approach:

* is robust to noisy or inconsistent judgments,
* avoids overfitting to a single judge decision,
* produces statistically grounded rankings.

### Why Bayesian Bradley–Terry?

LLM-as-a-Judge evaluations are inherently **noisy**:
the same judge model may produce inconsistent preferences across similar answer pairs,
and individual judgments can be affected by prompt phrasing, order effects, or randomness.

Instead of treating each judgment as a deterministic signal,
we adopt a **Bayesian Bradley–Terry (BT)** model for aggregation.

This choice provides several advantages:

- **Principled noise modeling**  
  Bayesian BT explicitly models uncertainty in pairwise preferences,
  allowing noisy or contradictory judgments to be absorbed probabilistically
  rather than dominating the final ranking.

- **Uncertainty-aware rankings**  
  The posterior distribution over model skill scores enables us to quantify
  confidence intervals and compare not only point estimates but also ranking stability.

- **Robustness to sparse or imbalanced comparisons**  
  In practice, not all model pairs are judged equally often.
  Bayesian inference naturally regularizes under sparse observations,
  avoiding brittle rankings caused by limited comparisons.

- **Better alignment with human and LLM judges**  
  Since both human and LLM judges are imperfect annotators,
  modeling preferences as stochastic variables is more realistic
  than assuming fully reliable pairwise outcomes.

Overall, Bayesian BT offers a statistically grounded and robust framework
for aggregating LLM-judge preferences, especially in high-noise evaluation settings.


#### Usage

```bash
python Bayes_bt.py \
  --answers_dir model_answers/ \
  --judge_model JUDGE_MODEL_NAME \
  --output results/
```

The script will:

1. sample answer pairs across models,
2. query the judge model for preferences,
3. fit a Bayesian BT model,
4. output rankings and posterior statistics.

Outputs include:

* model rankings,
* posterior means and variances,
* pairwise win probabilities.

---

## 4. Noise and Robustness Analysis

### Script: `noise_analysis.py`

This script evaluates **ranking stability under noise**, addressing a key concern of
LLM-as-a-Judge evaluations.

We analyze:

* how rankings change when judgment noise increases,
* sensitivity to judge inconsistency,
* robustness of top-k model ordering.

#### Usage

```bash
python noise_analysis.py \
  --results_dir results/ \
  --output analysis/
```

The analysis produces:

* Kendall’s τ / Spearman correlation under noise,
* top-k retention metrics,
* visualizations and summary statistics.

These experiments support the reliability claims of our evaluation framework.

---

## Reproducibility Notes

* All random seeds used in sampling and inference are logged.
* Intermediate artifacts (answers, judgments, posteriors) are saved for inspection.
* Each stage can be re-run independently.

---

## Acknowledgements

* Prompt curation is based on **BenchBuilder**.
* Ranking is inspired by classical **Bradley–Terry models**, extended with Bayesian inference.
* Evaluation philosophy aligns with recent LLM-as-a-Judge benchmarks (e.g., Arena-style evaluation).

