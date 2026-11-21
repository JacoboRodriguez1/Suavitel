# Sentiment Classification on GoEmotions

Integrative Task 2 - Computability and Data Structures III (CyED3), Universidad Icesi.

## Objective

This project builds a supervised binary sentiment classifier (positive / negative) for Reddit comments from the
[GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset (Demszky et al., 2020).
Each comment is labeled using the official sentiment mapping of the emotions, and a balanced dataset of
500 positive and 500 negative comments is built and split into train and test sets.

The following models are trained and compared:

- **DummyClassifier** (baseline)
- **Dense neural network**
- **Vanilla RNN**
- **LSTM**
- **DistilBERT** (Transformer extension)

Models are evaluated with accuracy, precision, recall, F1-score and Cohen's kappa. The architectures are also
discussed in relation to Turing machine concepts (memory, sequence processing and computability).

## Team members

- Nicolas Mejia
- Daniel Barragan
- Jacobo Rodriguez

## Repository structure

```
.
├── README.md
├── requirements.txt
├── environment.yml
├── data/
│   ├── raw/                 # original GoEmotions files (not versioned)
│   ├── processed/           # balanced dataset and train/test splits
│   └── README.md
├── notebooks/               # EDA, baseline, models and comparison
├── src/                     # reusable code (preprocessing, models, training, evaluation)
│   └── __init__.py
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── saved_models/
├── docs/                    # report and presentation
├── prompts/
│   ├── prompt_logs.txt
│   └── AIGen_Interactions.md
└── logs/
    ├── training_logs.txt
    └── experiment_notes.md
```

## Setup

Requires Python 3.11.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows (on Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

With conda:

```bash
conda env create -f environment.yml
conda activate goemotions-sentiment
```

## How to run

To be completed.

## Results

To be completed.
