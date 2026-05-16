# Benchmark Results

**Image:** test_cat.jpg  
**Warm-up:** 20 iterations  
**Measured:** 50 iterations  

| Model Type         |   Size (MB) |   Avg (ms) |   P50 (ms) |   P95 (ms) |   Min (ms) |   Max (ms) |
|:-------------------|------------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| Original (PyTorch) |       13.54 |      17.74 |      17.32 |      22.22 |      13.32 |      25.99 |
| ONNX               |        0.33 |       2.23 |       2.19 |       2.6  |       1.95 |       2.69 |
| Quantized ONNX     |        3.6  |      20.1  |      19.36 |      22.25 |      17.87 |      39.1  |
