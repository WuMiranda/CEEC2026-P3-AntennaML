# Inference Benchmark

- split: `test`
- num_samples: `19347`
- fp32 accuracy: `0.940921`
- int8 accuracy: `0.929653`
- fp32 macro_f1: `0.890938`
- int8 macro_f1: `0.879092`
- fp32 mean_ms: `0.037777`
- int8 mean_ms: `0.063522`
- speedup_vs_fp32: `0.594705`

## Method

- Latency is measured on CPU only.
- Input data uses the saved experiment split and saved preprocessing parameters.
- Each run measures a full forward pass of the final inference model, including prior-logit adjustment.
- Statistics report mean/std/min/max/p50/p95/p99 in milliseconds.
