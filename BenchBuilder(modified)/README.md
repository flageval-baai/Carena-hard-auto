# BenchBuilder (Customized for Our Project)

This repository contains a **customized version of BenchBuilder**, an automatic two-stage pipeline for curating high-quality benchmark prompts from large-scale, crowdsourced data.

Our implementation is based on the original **BenchBuilder** pipeline (Li294
et al., 2024) proposed in  
> *BenchBuilder: An Automatic Pipeline for Creating High-Quality Benchmarks*  
> https://arxiv.org/abs/2406.11939  

and follows the same directory structure and execution workflow.  
We introduce **lightweight but important modifications** to better support our research setting, as described in **Section 3.3 of our paper**.

> **Note**  
> This codebase is intended to reproduce the prompt curation pipeline used in our project.  
> While it can be applied to other datasets, some components (e.g., filtering rules and annotation criteria) are tailored to our benchmark construction.

---

## Overview

BenchBuilder employs a **two-stage LLM-based annotation and filtering pipeline**:

1. **Stage 1 (Filtering)**  
   - Use a relatively inexpensive LLM (e.g., GPT-3.5) to annotate prompt quality.
   - Filter out:
     - low-quality prompts, and
     - low-quality topic clusters.

2. **Stage 2 (High-Quality Selection)**  
   - Re-annotate remaining prompts with a stronger LLM (e.g., GPT-4).
   - Select only prompts and clusters that satisfy strict quality thresholds.

Our pipeline preserves this structure while adding **prompt validity checks, safety filtering, and rubric adjustments** motivated by our dataset and task.

---

## Installation

First, install the required dependencies:

```bash
cd BenchBuilder
pip install -r requirements.txt
````

Make sure you have properly configured access to the LLMs used for annotation (e.g., OpenAI API keys if using GPT models).

---

## Pipeline Usage

### 1. (Optional but Recommended) Prompt Pre-filtering

Before running BenchBuilder, we apply **lightweight preprocessing** to remove prompts that are unsuitable for reliable evaluation, including:

* prompts without a clear task or question (e.g., underspecified chit-chat), and
* prompts containing sensitive or privacy-related content.

This step improves the stability and interpretability of downstream LLM judgments.
Details of this filtering are described in **Section 3.3 of our paper**.

You may implement this step using simple heuristics or a lightweight classifier, depending on your dataset.

---

### 2. Topic Clustering

Cluster prompts into semantic topic groups:

```bash
python topic_clustering.py \
  --conv-file [your_prompts.json] \
  --min-topic-size 8
```

* Each prompt is embedded and grouped into a topic cluster.
* Clustering enables **cluster-level quality control** in later stages.

---

### 3. Stage-1 Annotation (Coarse Quality Scoring)

Annotate prompts using a relatively cheap LLM:

```bash
python label.py --config config.yaml
```

Before running, configure `config.yaml`, including:

* input prompt file,
* clustering file,
* LLM model name,
* annotation instructions and rubric.

The output contains **prompt-level quality scores** and will be used for initial filtering.

---

### 4. Stage-1 Filtering

Remove low-quality prompts and clusters:

```bash
python filter.py \
  --conversations_file [stage1_annotations.jsonl] \
  --clusters_file [clusters.json] \
  --prompt_threshold 5 \
  --cluster_threshold 3
```

This step:

* removes prompts with quality score < 5, and
* removes clusters with mean score < 3.

The goal is to cheaply discard obvious low-quality content before expensive annotation.

---

### 5. Stage-2 Annotation (High-Quality Scoring)

Re-annotate the filtered prompts using a stronger LLM:

```bash
python label.py --config config_stage2.yaml
```

Compared to Stage 1:

* a stronger model is used,
* annotation judgments are more reliable,
* the same rubric is applied with stricter interpretation.

---

### 6. Final Filtering

Select only high-quality prompts:

```bash
python filter.py \
  --conversations_file [stage2_annotations.jsonl] \
  --clusters_file [clusters.json] \
  --prompt_threshold 6 \
  --cluster_threshold 6
```

This yields a final set of **high-quality, challenging prompts** suitable for benchmarking.

---

## Key Differences from Original BenchBuilder

While we follow the original BenchBuilder workflow, we introduce several **task-driven adjustments**, motivated by our dataset and evaluation goals:

* **Prompt Validity Filtering**
  We remove prompts that lack clear evaluation criteria (e.g., vague or purely conversational inputs), which can lead to noisy or unstable LLM judgments.

* **Safety and Privacy Filtering**
  Prompts containing sensitive personal information or high-risk content are filtered out prior to annotation to ensure ethical and reproducible evaluation.

* **Rubric Adaptation**
  We adopt the original multi-dimensional quality rubric from BenchBuilder, with minor adjustments to better reflect our task setting, as detailed in the paper.

* **Threshold Calibration**
  Quality thresholds are tuned to balance recall and precision for our dataset, producing a compact but challenging benchmark.

These changes are **lightweight by design** and do not alter the core philosophy of BenchBuilder.

---

## Notes

* The final prompt set can be **stratified or sampled** by cluster to construct a benchmark.
* You may replace GPT-based annotators with other LLMs by modifying the configuration files.
* All intermediate files are saved in JSON/JSONL format for transparency and inspection.


## Reference(s)
Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica. 2024. From crowdsourced data to high-quality benchmarksArena-hard and benchbuilder pipeline. arXiv preprint arXiv:2406.11939.