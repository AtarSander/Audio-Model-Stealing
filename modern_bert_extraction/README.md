# Modern BERT Extraction

This directory contains a separate PyTorch/Hugging Face reimplementation of the
paper's classifier extraction pipeline. The goal is to preserve the original
paper logic while replacing the legacy TensorFlow 1.15 framework.

Current scope:

1. `SST-2`
2. `MNLI`
3. `RANDOM` and `WIKI` query-generation schemes
4. Victim fine-tuning, victim querying, soft-label distillation, extracted-model training
5. Dev-set accuracy and victim/extracted agreement metrics

Faithfulness to the original paper code:

1. The classifier tasks are still `SST-2` and `MNLI`.
2. Victim and extracted models both start from a pretrained BERT model.
3. `RANDOM` generation uses uniform random token sampling from the top-10k WikiText vocabulary.
4. `WIKI` generation samples WikiText sentences and sanitizes out-of-vocabulary tokens.
5. `MNLI` still uses the original `random_ed_k_uniform` idea: sample a premise and create the hypothesis by applying `k=3` token replacements.
6. Distillation still uses the original soft-label loss `-sum(p_teacher * log p_student)`.

Suggested first run:

```bash
uv run -m modern_bert_extraction.run_classifier_pipeline \
  --config modern_bert_extraction/configs/classifier.yaml \
  --task SST-2 \
  --scheme random
```
