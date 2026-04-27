# 仅更新计时函数（4.25/wyx)

import time
def time_inference(model, x_numpy, warmup=10, runs=100):
    model.eval()
    x_tensor = torch.from_numpy(x_numpy).unsqueeze(0)  # (1, N)
    
    # 冷启动耗时（模型加载 + 首次推理）
    t_load = time.perf_counter()
    with torch.no_grad():
        _ = model(x_tensor)
    cold_start = time.perf_counter() - t_load
    
    # 稳定推理耗时
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x_tensor)
        times.append(time.perf_counter() - t0)
    
    return {
        "cold_start_ms": cold_start * 1000,
        "avg_infer_ms": np.mean(times) * 1000,
        "p95_infer_ms": np.percentile(times, 95) * 1000
    }