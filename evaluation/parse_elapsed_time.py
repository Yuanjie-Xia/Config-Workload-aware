#!/usr/bin/env python3
import re, sys
from decimal import Decimal, getcontext
from datetime import timedelta
from pathlib import Path

# -------- settings --------
LOGFILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../results/rq1/trainticket_mix/qpme_XG_model/n/train.0")
getcontext().prec = 28  # high precision summation
# --------------------------

# number pattern like 12, 12.3, .5, 1e-3
NUM = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"

# Patterns
re_elapsed = re.compile(rf"\bElapsed time:\s*{NUM}\b")
# Accepts: "Request response time: 123", "Request response time: 123 ms", "… 0.25 s"
re_req = re.compile(rf"\bRequest response time:\s*{NUM}\s*(ms|s|sec|secs|second|seconds)?\b", re.IGNORECASE)

elapsed_total = Decimal("0")
elapsed_count = 0

req_total_seconds = Decimal("0")
req_count = 0

with LOGFILE.open("r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()

        m = re_elapsed.search(line)
        if m:
            elapsed_total += Decimal(m.group(1))
            elapsed_count += 1

        m = re_req.search(line)
        if m:
            val = Decimal(m.group(1))
            unit = (m.group(2) or "").lower()
            # Default = seconds if no unit; convert ms -> seconds
            if unit in ("ms",):
                val = val / Decimal("1000")
            # treat sec variants as seconds; else assume seconds
            req_total_seconds += val
            req_count += 1

def fmt_td(total_seconds: Decimal) -> str:
    return str(timedelta(seconds=float(total_seconds)))

print(f"File: {LOGFILE}")

print(f"\nElapsed time entries: {elapsed_count}")
print(f"  Total seconds: {elapsed_total}")
print(f"  Total duration: {fmt_td(elapsed_total)}")
if elapsed_count:
    avg = elapsed_total / elapsed_count
    print(f"  Mean per entry (s): {avg}")

print(f"\nRequest response time entries: {req_count}")
print(f"  Total seconds: {req_total_seconds}")
print(f"  Total duration: {fmt_td(req_total_seconds)}")
if req_count:
    avg = req_total_seconds / req_count
    print(f"  Mean per entry (s): {avg}")
