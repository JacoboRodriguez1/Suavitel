# Experiment Notes

Notes on data checks, design decisions and experiment results.

## Template

### YYYY-MM-DD - Title

- **Author:**
- **Goal:**
- **Setup:** (data, model, hyperparameters, seed)
- **Result:**
- **Conclusion / next step:**

## Notes

### 2026-11-21 - Sentiment mapping check

- **Author:** Nicolas Mejia
- **Goal:** assign a sentiment to each GoEmotions comment with the official `sentiment_mapping.json` and check the counts expected by the assignment.
- **Setup:** filtered version (train + dev + test = 54,263 rows; 43,410 / 5,426 / 5,427), TSV files read with `quoting=csv.QUOTE_NONE`. Each emotion is mapped to positive / negative / ambiguous, and neutral is its own group.
- **Rule that matches:** if all the emotions of a comment fall in one group, that group is the sentiment; otherwise the comment is mixed (neutral together with any other emotion also counts as mixed).
- **Result:** positive 19,020, negative 10,217, ambiguous 4,805, neutral 16,021, mixed 4,200 (total 54,263): exact match, enforced by `check_sentiment_counts()`.
- **Rejected variant:** ignoring neutral when it appears with other emotions gives positive 19,763, negative 10,819, ambiguous 5,190, neutral 16,021, mixed 2,470, which does not match. 1,751 comments have neutral plus another emotion.
- **Conclusion / next step:** only positive and negative comments go on to the balanced dataset.
