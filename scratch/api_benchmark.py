import requests
import time
import numpy as np

def benchmark_api(url, image_path, runs=50):
    print(f"Benchmarking API at {url} using {image_path}...")
    files = {'file': open(image_path, 'rb')}
    
    # Warmup
    try:
        requests.post(url, files={'file': open(image_path, 'rb')})
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return

    latencies = []
    for i in range(runs):
        start = time.perf_counter()
        response = requests.post(url, files={'file': open(image_path, 'rb')})
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        if i % 10 == 0:
            print(f"  Run {i}/{runs}...")
    
    avg = np.mean(latencies)
    p95 = np.percentile(latencies, 95)
    print(f"\n### API Benchmark Summary")
    print(f"| Metric | Value |")
    print(f"|---|---|")
    print(f"| Avg Latency | {avg:.2f} ms |")
    print(f"| P95 Latency | {p95:.2f} ms |")
    print(f"| Total Runs | {runs} |")

if __name__ == "__main__":
    benchmark_api("http://localhost:8000/predict", "test_cat.jpg")
