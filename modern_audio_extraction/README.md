# Modern Audio Encoder Extraction

This package implements the second-stage audio experiment: stealing a pretrained
self-supervised audio encoder, with HuBERT as the default target.

The implementation keeps the StolenEncoder logic and changes only the framework
and modality:

1. Load a target encoder such as `facebook/hubert-base-ls960`.
2. Query it once on a surrogate audio set and cache `f_target(x)`.
3. Train a stolen encoder with `L = d(f_target(x), f_stolen(x)) + lambda * d(f_target(x), f_stolen(aug(x)))`.
4. Compare the stolen encoder to the target by feature MSE/cosine similarity.
5. Optionally train equal downstream probes on target and stolen embeddings and compare accuracy.

Default config:

```text
modern_audio_extraction/configs/hubert_stolenencoder.yaml
```

Run a full HuBERT experiment:

```bash
make modern_audio_hubert
```

Run a small synthetic smoke test:

```bash
make modern_audio_hubert_smoke
```

Run the query-budget / surrogate-architecture / query-source matrix:

```bash
make modern_audio_hubert_matrix
```

Collect text and audio results into one comparison table:

```bash
make modern_collect_results
```

Outputs are written under:

```text
output/repro/modern_audio/
output/repro/comparison/
checkpoints/repro/modern_audio/
```

The main result files are:

```text
cache/target_feature_cache.pt
metrics/distillation_metrics.json
metrics/feature_similarity.json
metrics/downstream_metrics.json
metrics/final_metrics.json
checkpoints/repro/modern_audio/<run-name>/stolen_encoder/
```
