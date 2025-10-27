import re
import statistics
from pathlib import Path


def mean_stddev_from_screenlog(path="screenlog.0"):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")

    # Match floats like 0.1492, -0.1, 1e-3, 2.3E+05 after "Standard Deviation:"
    pattern = re.compile(r"Standard\s+Deviation:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    vals = [float(x) for x in pattern.findall(text)]

    if not vals:
        print("No 'Standard Deviation:' entries found.")
        return

    print(f"Entries found: {len(vals)}")
    print(f"Mean of Standard Deviation: {statistics.fmean(vals):.12f}")  # fmean is fast & precise
    # If you also want min/max:
    # print(f"Min: {min(vals):.12f}, Max: {max(vals):.12f}")


if __name__ == "__main__":
    mean_stddev_from_screenlog("../results/rq1/trainticket/only_xg_model/train.0")