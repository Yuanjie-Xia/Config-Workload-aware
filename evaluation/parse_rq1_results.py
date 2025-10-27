import ast
import re
from statistics import mean

path = "../results/rq1/teastore_mix/qpme_XG_model/5n/dup.0"  # change if needed
prefix_error = "Relative_error: "
prefix_spearcor = "Spearman correlation: "
length = 114
means = []
cor_means = []
length_fit = False
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for lineno, line in enumerate(f, 1):
        if line.startswith(prefix_error):
            # grab the substring after the prefix
            payload = line[len(prefix_error):].strip()
            try:
                # parse Python list literal safely
                arr = ast.literal_eval(payload)
            except Exception as e:
                print(f"[line {lineno}] parse error: {e}")
                continue

            if isinstance(arr, list) and len(arr) == length:
                length_fit = True
                try:
                    m = mean(float(x) for x in arr)
                    means.append(m)
                    print(f"[line {lineno}] mean = {m:.12f}")
                except Exception as e:
                    print(f"[line {lineno}] numeric error: {e}")
            else:
                # skip lists not length 18
                pass
        if line.startswith(prefix_spearcor):
            # grab the substring after the prefix
            payload = line[len(prefix_spearcor):].strip()
            try:
                # parse Python list literal safely
                arr = float(payload)
            except Exception as e:
                print(f"[line {lineno}] parse error: {e}")
                continue

            if isinstance(arr, float) and length_fit:
                length_fit = False
                try:
                    cor_means.append(arr)
                    print(f"[line {lineno}] cor = {arr:.12f}")
                except Exception as e:
                    print(f"[line {lineno}] numeric error: {e}")
            else:
                # skip lists not length 18
                pass
print(cor_means)
print(means)
if means:
    print(f"\nProcessed {len(means)} lines.")
    print(f"Overall average of per-line means = {mean(means):.12f}")
    print(f"\nProcessed {len(cor_means)} lines.")
    print(f"Overall average of per-line mean cor = {mean(cor_means):.12f}")
else:
    print("No length-18 lists found after 'Relative_error: '.")