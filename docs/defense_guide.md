# Technical defense notes

Use these notes to prepare evidence-based answers. Keep the limits of this experiment clear: the results come from one balanced 200-comment test split, and the model means summarize different numbers of seeds.

## Dataset and evaluation

**Why convert GoEmotions into positive and negative sentiment?** The official mapping groups the emotion labels into positive, negative and ambiguous categories. This experiment keeps only comments whose labels all map to one of the positive or negative groups. It excludes neutral, ambiguous and mixed comments, removes duplicate text, and samples 500 comments per class with seed 42.

**How did the evaluation avoid test leakage?** The 800-comment training split was divided into 680 fit and 120 validation examples. Vectorization vocabulary and model selection used training and validation only. The same held-out 200 comments were reserved for final testing across models. DistilBERT's learning rate and checkpoints were selected by validation macro F1.

**Why report macro F1 and Cohen's kappa as well as accuracy?** Accuracy summarizes total correct predictions. Macro F1 gives the two sentiment classes equal weight, and kappa measures agreement adjusted for chance. The balanced test split helps comparison, but reporting several metrics makes class-specific behavior clearer.

## Models and results

**What changes across the architectures?** The Dummy baseline predicts from class frequencies and does not read text. Dense averages token embeddings, which is efficient but discards word order. SimpleRNN updates a hidden state token by token. LSTM adds gated cell memory. DistilBERT uses pretrained subword representations and self-attention over the original comment.

**Why did the models perform differently?** Mean test macro F1 was 0.516 for the stratified Dummy, 0.741 for Dense, 0.551 for SimpleRNN, 0.701 for LSTM and 0.843 for DistilBERT. This experiment shows the ranking on this sample; it does not establish that the same ranking will hold for a larger or different source of comments. The 800 training comments and 30.7% test-token OOV rate for the recurrent/Dense vocabulary are important limitations.

**What do the training and test gaps indicate?** Dense's mean macro-F1 gap was 0.249, compared with 0.093 for DistilBERT. Every neural model scored better on training than test data. This is evidence of a generalization gap on the fixed split, not proof that model size alone caused it.

## Computational cost and interpretability

**How does training cost compare?** On the recorded CPU environment, one final refit averaged 0.855 s for Dense, 1.301 s for SimpleRNN, 1.506 s for LSTM and 644.881 s for DistilBERT. These times exclude the hyperparameter search and depend on the machine and software. DistilBERT used three seeds; the other models used five. The comparison measured no GPU timing.

**What is the inference trade-off?** For 1,000 already-tokenized comments, CPU inference took 0.184 s for Dense, 0.185 s for SimpleRNN, 0.221 s for LSTM and 47.193 s for DistilBERT. A deployment decision depends on whether its accuracy gain justifies its latency and resource needs. These measurements exclude cleaning and tokenization.

**Which model is easiest to interpret?** The Dummy baseline has a transparent prediction rule but ignores comment content. Dense has a simple mean-pooling path, although it loses order. RNN and LSTM store information in learned hidden states that are harder to inspect. DistilBERT's attention weights can be visualized, but they show where attention is allocated and do not prove why a prediction happened.

**What does computational complexity mean here?** For sequence length `T`, embedding width `d` and recurrent width `h`, Dense mean-pooling uses `O(Td)` operations, SimpleRNN uses `O(T(dh + h²))` sequential operations, and LSTM has the same order with a larger constant for its gates. DistilBERT's self-attention term is `O(T²d)` per layer and processes positions in parallel within a bounded context. These asymptotic terms explain scaling; the measured fit and inference times show this implementation on one CPU.

## Computability and limitations

**Does a Turing-completeness result make these classifiers universal computers?** No. Theoretical results for recurrent networks or Transformers rely on idealized assumptions. The trained classifiers use fixed parameters, finite precision and bounded input lengths. Each inference performs a finite sequence of operations and terminates with a label.

**What should be done next?** Evaluate on a larger, independently sampled test set and another comment source. Measure inference on target deployment hardware, including a GPU if relevant. Inspect errors involving negation, irony and emoji; the current attention plots are qualitative diagnostics, not causal explanations.

## Evidence to have ready

- `docs/report.pdf` for the methods, full train/test metrics, confusion matrices and references.
- `outputs/metrics/comparison.csv` for per-model means, standard deviations, parameter counts, fit times and inference times.
- `outputs/metrics/distilbert_lr_search.csv` for validation-only learning-rate selection.
- `logs/experiment_notes.md` for seeds, preprocessing checks, fit protocol and reproducibility details.
- `notebooks/04_transformer_extension.ipynb` for the Transformer setup, attention inspection and comparison workflow.
