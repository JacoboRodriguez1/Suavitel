# Generative AI Interactions

Record of how generative AI tools were used during the project, what was accepted and what was changed.

## 2026-11-21 - Nicolas Mejia - Repository setup and dependencies

- **Tool:** AI assistant (LLM).
- **Request:** propose the folder structure required by the assignment, `requirements.txt` and `environment.yml` for Python 3.11 on Windows, a `.gitignore`, and initial `README.md` / `data/README.md` stubs.
- **Review and adjustments:** checked the structure against the assignment statement; verified with pip that the chosen version ranges resolve on Python 3.11 / Windows (TensorFlow 2.20 is the newest version inside the `<2.21` range) and installed the lightweight packages in a local virtual environment; made the `.gitignore` keep `data/raw/.gitkeep` while ignoring the raw data files; README sections "How to run" and "Results" left as placeholders to be completed by the team.

## 2026-11-21 - Nicolas Mejia - Data loading and official sentiment mapping

- **Tool:** AI assistant (LLM).
- **Request:** functions in `src/preprocessing.py` to download and load the filtered GoEmotions files, read the official `sentiment_mapping.json` and assign a sentiment to each comment; documentation of the source, citation and counts.
- **Review and adjustments:** the Kaggle download needs a login, so the same filtered files were taken from the authors' repository and this is stated in `data/README.md`. Checked that the three TSV files load to exactly 54,263 rows with quoting disabled. Tested two rules for neutral combined with other emotions; only "neutral is its own group, any mix of groups is mixed" reproduces the expected counts (19,020 / 10,217 / 4,805 / 16,021 / 4,200), and the other rule was rejected. Added an assert (`check_sentiment_counts`) so the pipeline stops if the counts change. Checked the emotion order against `emotions.txt`.
