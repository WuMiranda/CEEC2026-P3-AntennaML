# Model Profile

- model_dir: `outputs/expN_minusplus`
- input_dim: `24`
- num_classes: `15`
- hidden_dims: `[64, 32]`
- fp32 params: `4367`
- fp32 MACs: `4064`
- fp32 FLOPs: `8719`
- fp32 size(MB): `0.022494`
- int8 size(MB): `0.012773`

## Method

- params: Counts all learnable parameters in the saved PyTorch model.
- macs: Counts only Linear-layer multiply-accumulate operations for single-sample forward inference (batch=1).
- flops: Counts Linear FLOPs as 2*MACs plus bias adds, and also includes BatchNorm1d/ReLU elementwise inference FLOPs.
