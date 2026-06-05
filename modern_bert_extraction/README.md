# Modern BERT Extraction

This directory contains a separate PyTorch/Hugging Face reimplementation of the
paper's classifier and QA extraction pipelines. The goal is to preserve the
original paper logic while replacing the legacy TensorFlow 1.15 framework.

Current scope:

1. `SST-2`
2. `MNLI`
3. `SQuAD 1.1`
4. `BoolQ`
5. `RANDOM` and `WIKI` query-generation schemes
6. Victim fine-tuning, victim querying, distillation, extracted-model training
7. Dev-set task metrics and victim/extracted agreement metrics

Faithfulness to the original paper code:

1. The classifier tasks are still `SST-2` and `MNLI`.
2. Victim and extracted models both start from a pretrained BERT model.
3. `RANDOM` generation uses uniform random token sampling from the top-10k WikiText vocabulary.
4. `WIKI` generation samples WikiText sentences and sanitizes out-of-vocabulary tokens.
5. `MNLI` still uses the original `random_ed_k_uniform` idea: sample a premise and create the hypothesis by applying `k=3` token replacements.
6. Distillation still uses the original soft-label loss `-sum(p_teacher * log p_student)`.
7. SQuAD/BoolQ query generation uses WikiText paragraphs, paper-style random question sampling, and the paper's task-specific question starters.

Suggested classifier run:

```bash
uv run -m modern_bert_extraction.run_classifier_pipeline \
  run.task=SST-2 \
  run.scheme=random
```

Suggested QA runs:

```bash
uv run -m modern_bert_extraction.run_squad_pipeline \
  run.scheme=random

uv run -m modern_bert_extraction.run_boolq_pipeline \
  run.scheme=random
```

The runners are Hydra entrypoints. Use `run.step=<stage>` for partial runs and
override config values with dotlist syntax, for example:

```bash
uv run -m modern_bert_extraction.run_classifier_pipeline \
  run.task=MNLI \
  run.scheme=wiki \
  training.per_device_train_batch_size=4
```

The top-level configs are composition files. Reusable blocks live under:

```text
configs/model/
configs/paths/
configs/training/
configs/query_generation/
configs/runtime/
```

Inspect the resolved config without launching a run:

```bash
uv run -m modern_bert_extraction.run_classifier_pipeline --cfg job
```

Suggested WIKI paper-comparison run after RANDOM victims exist:

```bash
make modern_reproduce_wiki_reuse_victims
```

The reuse targets override `paths.victim_model_dir`, so WIKI extraction uses
the same victim checkpoints as the RANDOM runs for each task.
