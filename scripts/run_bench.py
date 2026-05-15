import subprocess
import os
import time
import csv
import re

def run_cmd(cmd):
    print(f"Running: {cmd}")
    start = time.time()
    try:
        # We use /usr/bin/time -p to get portable real time
        result = subprocess.run(f"/usr/bin/time -p {cmd}", shell=True, capture_output=True, text=True)
        # Extract real time from stderr
        match = re.search(r'real\s+([0-9.]+)', result.stderr)
        real_time = float(match.group(1)) if match else (time.time() - start)
        return real_time
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    steps = 120
    binary = "./target/release/train_generic"
    
    # Matrix A
    matrix_a_results = []
    os.makedirs("runs/bench_matrix_a", exist_ok=True)
    for b in [4, 8, 16]:
        for s in [32, 64, 128]:
            out_dir = f"runs/bench_matrix_a/b{b}_s{s}"
            cmd = f"{binary} --steps {steps} --batch-size {b} --seq-len {s} --d-model 512 --d-state 16 --base-layers 2 --target-layers 2 --out-dir {out_dir}"
            real_s = run_cmd(cmd)
            if real_s:
                step_s = steps / real_s
                token_s = (steps * b * s) / real_s
                matrix_a_results.append({
                    "config": f"b{b}_s{s}_d512_l2", "batch": b, "seq": s, "d_model": 512, "layers": 2,
                    "real_s": round(real_s, 2), "step_s": round(step_s, 2), "token_s": round(token_s, 2)
                })

    matrix_a_results.sort(key=lambda x: x['token_s'], reverse=True)
    with open("runs/bench_matrix_a/results.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=matrix_a_results[0].keys())
        writer.writeheader()
        writer.writerows(matrix_a_results)

    # Matrix B
    matrix_b_results = []
    os.makedirs("runs/bench_matrix_b", exist_ok=True)
    for dm in [256, 512]:
        for l in [2, 4, 6, 8, 10]:
            out_dir = f"runs/bench_matrix_b/dm{dm}_l{l}"
            cmd = f"{binary} --steps {steps} --batch-size 8 --seq-len 64 --d-model {dm} --d-state 16 --base-layers {l} --target-layers {l} --out-dir {out_dir}"
            real_s = run_cmd(cmd)
            if real_s:
                step_s = steps / real_s
                token_s = (steps * 8 * 64) / real_s
                matrix_b_results.append({
                    "config": f"b8_s64_d{dm}_l{l}", "batch": 8, "seq": 64, "d_model": dm, "layers": l,
                    "real_s": round(real_s, 2), "step_s": round(step_s, 2), "token_s": round(token_s, 2)
                })

    matrix_b_results.sort(key=lambda x: x['token_s'], reverse=True)
    with open("runs/bench_matrix_b/results.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=matrix_b_results[0].keys())
        writer.writeheader()
        writer.writerows(matrix_b_results)

    print("\nMatrix A Results:")
    print("config,batch,seq,d_model,layers,real_s,step_s,token_s")
    for r in matrix_a_results:
        print(f"{r['config']},{r['batch']},{r['seq']},{r['d_model']},{r['layers']},{r['real_s']:.2f},{r['step_s']:.2f},{r['token_s']:.2f}")

    print("\nMatrix B Results:")
    print("config,batch,seq,d_model,layers,real_s,step_s,token_s")
    for r in matrix_b_results:
        print(f"{r['config']},{r['batch']},{r['seq']},{r['d_model']},{r['layers']},{r['real_s']:.2f},{r['step_s']:.2f},{r['token_s']:.2f}")

    print(f"\nBest Matrix A: {matrix_a_results[0]['config']} ({matrix_a_results[0]['token_s']:.2f} tokens/s)")
    print(f"Best Matrix B: {matrix_b_results[0]['config']} ({matrix_b_results[0]['token_s']:.2f} tokens/s)")

if __name__ == "__main__":
    main()
