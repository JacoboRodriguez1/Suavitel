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
│   ├── report/              # LaTeX source of the report (main.tex + sections/)
│   └── references.bib
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

All commands are run from the repository root with the virtual environment active.

1. **Data.** The processed files in `data/processed/` (`balanced.csv`, `train.csv`, `test.csv`) are already in the
   repository. To regenerate them from scratch (downloads the raw GoEmotions files into `data/raw/` if needed):

   ```bash
   python -c "from src.preprocessing import download_raw; download_raw()"
   python -m src.preprocessing
   ```

   See `data/README.md` for the sentiment mapping, the balancing, the cleaning and the train/test split.

2. **Notebooks.** Run them in order, from the `notebooks/` folder (they import the code in `src/` with
   `sys.path.append("..")`), either in Jupyter or from the command line:

   ```bash
   cd notebooks
   jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
   jupyter nbconvert --to notebook --execute --inplace 02_baseline_model.ipynb
   python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 03_dense_rnn_lstm.ipynb
   ```

   | Notebook | Content | Outputs |
   |---|---|---|
   | `01_eda.ipynb` | Data quality, sentiment mapping, balanced dataset, sequence length, preprocessing and vocabulary | `outputs/figures/eda_*.png` |
   | `02_baseline_model.ipynb` | `DummyClassifier` baselines (`most_frequent`, `stratified`, `uniform`) on train and test | `outputs/metrics/dummy.csv`, `outputs/figures/baseline_*.png` |
   | `03_dense_rnn_lstm.ipynb` | Tune Dense, SimpleRNN and LSTM, refit with five seeds and inspect test errors | Three tuning grids, three final metrics CSVs, six figures and three Keras models under `outputs/` |

   Notebook 01 also needs the five raw GoEmotions files listed in `data/README.md` and the NLTK English stopwords corpus. With internet access, the project can download these on first use. Notebooks 02 and 03 use the versioned processed data. Select the project's Python 3.11 environment when running interactively, then restart the kernel and run all cells in order.

   Notebook 03 uses 680 comments for fitting and 120 for validation. It ranks 48 settings per architecture by validation macro F1, using validation loss to break ties. With the winning settings, it selects the epoch count for each seed (42–46), refits on all 800 training comments and scores all five runs on the same 200 test comments. Each final metrics CSV has ten rows (train and test for each seed). Runtime measurements depend on the machine.

   The reference mean test macro F1 is 0.7408 for Dense, 0.5507 for SimpleRNN and 0.7006 for LSTM. Compare these with the five-seed summary in the notebook. See [the experiment notes](logs/experiment_notes.md) for validation scores, seed-specific results and the reproduction environment.

   To reproduce just the test matrices and error examples, run the setup cell followed by **Test confusion matrices and recurrent-model errors**. This uses the committed `outputs/saved_models/{dense,rnn,lstm}.keras` checkpoints and the token IDs in `data/processed/vocab.txt`. A full run rewrites the generated outputs and appends the training trials to `logs/training_logs.txt`.

## Results

To be completed.
