# --- config ---
LOGFILE = "../results/rq1/teastore_mix/full_dp_model/screenlog.0"         # path to your log file
TARGET_ROUNDS = {38, 76, 114, 152, 190}            # e.g., {90} or {45, 90}; set to None to include all rounds
# ---------------

import re
from pathlib import Path
from statistics import mean

ROUND_RE     = re.compile(r"^Round\s+(\d+):.*?Performance=([0-9.+-eE]+)")
NEXT_ROUND   = re.compile(r"^Round\s+\d+:")
SPEARMAN_RE  = re.compile(r"^\[10-Fold Rank Eval\]\s+Avg Spearman's Rank Correlation:\s*([0-9.+-eE]+)")
MRE_RE       = re.compile(r"^\[10-Fold CV\].*?Mean Relative Error:\s*([0-9.+-eE]+)")

def parse_rounds(path: str, target_rounds: set[int] | None):
    """Return {round_num: [ {'perf':float|None, 'spearman':float|None, 'mre':float|None}, ... ]}"""
    results: dict[int, list[dict]] = {}
    current = None
    current_round = None

    def push():
        if current_round is None or current is None:
            return
        if (target_rounds is None) or (current_round in target_rounds):
            results.setdefault(current_round, []).append(current)

    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            m = ROUND_RE.match(line)
            if m:
                # new block
                push()
                current_round = int(m.group(1))
                perf = float(m.group(2))
                current = {'perf': perf, 'spearman': None, 'mre': None}
                continue

            if current is not None and NEXT_ROUND.match(line):
                # terminate previous block (even if we don't care about next round)
                push()
                current = None
                current_round = None
                # let the next loop iteration catch the new Round line
                continue

            if current is not None:
                m = SPEARMAN_RE.match(line)
                if m:
                    current['spearman'] = float(m.group(1)); continue
                m = MRE_RE.match(line)
                if m:
                    current['mre'] = float(m.group(1)); continue

    # end of file
    if current is not None:
        push()

    return results

def print_report(results: dict[int, list[dict]]):
    if not results:
        print("No matching rounds found."); return

    for rnd in sorted(results):
        rows = results[rnd]
        for idx, row in enumerate(rows, 1):
            p  = f"{row['perf']:.4f}"      if row['perf']      is not None else "NA"
            sp = f"{row['spearman']:.4f}" if row['spearman']  is not None else "NA"
            mr = f"{row['mre']:.4f}"      if row['mre']       is not None else "NA"
            occ = f" (occurrence {idx})" if len(rows) > 1 else ""
            print(f"Round {rnd}{occ}: Performance={p}, Avg Spearman={sp}, Mean Rel Err={mr}")

        sp_vals = [r['spearman'] for r in rows if r['spearman'] is not None]
        mr_vals = [r['mre']      for r in rows if r['mre']      is not None]
        if sp_vals:
            print(f"Round {rnd} — Mean Avg Spearman: {mean(sp_vals):.4f} (n={len(sp_vals)})")
        if mr_vals:
            print(f"Round {rnd} — Mean Mean Relative Error: {mean(mr_vals):.4f} (n={len(mr_vals)})")
        print("-" * 60)


if __name__ == "__main__":
    res = parse_rounds(LOGFILE, TARGET_ROUNDS)
    print_report(res)
