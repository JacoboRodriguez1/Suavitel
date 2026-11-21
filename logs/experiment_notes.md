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

### 2026-11-21 - Balanced dataset and stratified split

- **Author:** Nicolas Mejia
- **Goal:** build the balanced 500/500 positive/negative dataset and the 80/20 train/test split.
- **Setup:** train + dev + test joined (54,263 rows), only positive and negative kept, duplicates removed on `text.strip().lower()` (repeated text with different sentiments: all copies dropped; otherwise keep the first), 500 per class sampled with `SEED = 42`, stratified `train_test_split` (test_size=0.2, random_state=42).
- **Candidates:** positive 19,020 -> 18,833 and negative 10,217 -> 10,188 after deduplication (29,237 -> 29,021).
- **Duplicates:** 124 repeated texts; 213 exact copies removed (mostly short comments such as "thank you!", "happy cake day!"); 1 conflicting text ("Weird flex but ok": negative twice, positive once) removed with its 3 copies.
- **Result:** balanced.csv 1,000 rows (500/500), train.csv 800 (400/400), test.csv 200 (100/100), no repeated texts and no comment_id shared between train and test (checked by `check_processed()`). Comment length in the balanced set: 1 to 29 words, mean 13.1.
- **Note:** deduplication only looks at positive/negative comments. One text of the balanced set ("YES!!!!", positive) also appears elsewhere in the corpus as neutral and as mixed; it was kept because those copies are not candidates.
- **Conclusion / next step:** `clean_text` is still a copy of `text`; text cleaning and the stopword policy come next.
