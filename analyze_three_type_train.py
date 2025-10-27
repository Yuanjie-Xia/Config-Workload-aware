import pickle
import json
from math import isfinite

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict,OrderedDict
import xml.etree.ElementTree as ET
import platform
import matplotlib
# Choose backend based on OS
if platform.system() == 'Darwin':  # macOS
    matplotlib.use('TkAgg')
elif platform.system() == 'Windows':
    matplotlib.use('Qt5Agg')
elif platform.system() == 'Linux':
    matplotlib.use('Agg')  # headless (e.g., servers)
else:
    matplotlib.use('TkAgg')  # default fallback

import matplotlib.pyplot as plt


def parse_probs(analysis_result_file, url):
    try:
        tree = ET.parse(analysis_result_file)
        root = tree.getroot()
        meanST = 0
        for observed in root.findall(".//observed-element"):
            if observed.attrib.get("type") == "probe":
                element_name = observed.attrib.get("name")
                if element_name != url:
                    continue
                for color in observed.findall("color"):
                    for metric in color.findall("metric"):
                        if metric.attrib.get("type") == "meanST":
                            meanST = float(metric.attrib["value"])
                            # print(f"[QPME] {element_name}: meanST = {meanST:.6f}")
    except Exception as e:
        print(f"[ERROR] Failed to parse analysis file: {e}")
        return []
    return meanST


def spearman_matrix(lists):
    # all unique items across lists
    items = lists[0]  # assuming each list is permutation of same items

    # map each list to ranks
    rank_vectors = []
    for lst in lists:
        rank_pos = {e: i for i, e in enumerate(lst)}
        rank_vectors.append([rank_pos[it] for it in items])

    rank_vectors = np.array(rank_vectors)

    n = len(lists)
    corr = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            rho, _ = spearmanr(rank_vectors[i], rank_vectors[j])
            corr[i,j] = rho
    return pd.DataFrame(corr, index=[f"list_{i}" for i in range(n)],
                               columns=[f"list_{i}" for i in range(n)])


def detect_spikes_neighbors(y, k=3.0, window=10, eps=1e-12):
    """
    A spike at t is defined as:
        (y[t] - median(neighbors)) / (1.4826 * MAD(neighbors)) > k
    where neighbors are values within `window` steps before and after t (excluding t).
    Returns: (indices, values)
    """
    a = np.asarray(y, dtype=float)
    n = len(a)
    spikes_idx, spikes_val = [], []
    for t in range(window, n - window):
        curr_ = a[t]
        if not np.isfinite(curr_):
            continue
        neighbors = np.concatenate([a[t-window:t], a[t+1:t+window+1]])
        neighbors = neighbors[np.isfinite(neighbors)]
        if len(neighbors) == 0:
            continue

        med = float(np.median(neighbors))
        mad = float(np.median(np.abs(neighbors - med)))
        scale = 1.4826 * mad
        scale = max(scale, eps)

        z = (curr_ - med) / scale
        rel = abs(curr_ - med) / med
        if z > k and rel >= 0.25:
            spikes_idx.append(t)
            spikes_val.append(curr_)
    return spikes_idx, spikes_val


# ---------- helpers to diff change_elements ----------
def _to_map(change_elements_list):
    """
    change_elements_list: e.g. [{'element': 'W/index','mean':'1.23','std':'0.01'}, ...]
    return dict: name -> {'mean': float or None, 'std': float or None, 'frequency': maybe str}
    """
    m = {}
    for d in change_elements_list:
        name = d.get('element')
        if not name:
            continue
        entry = {}
        # parse floats if present
        mean_s = d.get('mean')
        std_s = d.get('std')
        entry['mean'] = float(mean_s) if mean_s is not None else None
        entry['std'] = float(std_s) if std_s is not None else None
        # keep frequency as raw (often string)
        if 'frequency' in d:
            entry['frequency'] = d['frequency']
        m[name] = entry
    return m


def _fmt_float(x, nd=4):
    if x is None or not isfinite(x):
        return "NA"
    return f"{x:.{nd}f}"


def _diff_means(prev_map, curr_map, next_map, tol=1e-9):
    """
    Return list of (element, prev_mean, curr_mean, next_mean, d_prev, d_next)
    for elements where curr_mean differs from prev or next by > tol.
    """
    elements = set(curr_map.keys()) | set(prev_map.keys()) | set(next_map.keys())
    changes = []
    for e in sorted(elements):
        pm = prev_map.get(e, {}).get('mean')
        cm = curr_map.get(e, {}).get('mean')
        nm = next_map.get(e, {}).get('mean')

        # only consider if curr exists and is finite
        if cm is None or not isfinite(cm):
            continue

        # compare with prev/next if available
        diff_prev = (pm is None) or (not isfinite(pm)) or (abs(cm - pm) > tol)
        diff_next = (nm is None) or (not isfinite(nm)) or (abs(cm - nm) > tol)
        if diff_prev or diff_next:
            d_prev = (cm - pm) if (pm is not None and isfinite(pm)) else float('nan')
            d_next = (cm - nm) if (nm is not None and isfinite(nm)) else float('nan')
            changes.append((e, pm, cm, nm, d_prev, d_next))
    return changes


def _maybe_freq_change(prev_map, curr_map, next_map):
    out = []
    for key in ["Workload"]:
        if key in curr_map or key in prev_map or key in next_map:
            pf = prev_map.get(key, {}).get('frequency')
            cf = curr_map.get(key, {}).get('frequency')
            nf = next_map.get(key, {}).get('frequency')
            if cf != pf or cf != nf:
                out.append((key, pf, cf, nf))
    return out


def _finite_rows(arr):
    """Indices of rows with all finite perf values (exclude saturation)."""
    a = np.asarray(arr, dtype=float)
    mask = np.all(np.isfinite(a), axis=1)
    return np.where(mask)[0]


def _context_key_without(elem_name, ce_map, mean_round_decimals=6, keep_std=True, keep_freq=True):
    """Hashable key for 'all change_elements except elem_name'."""
    items = []
    for k in sorted(ce_map.keys()):
        if k == elem_name:
            continue
        v = ce_map[k]
        mean_ = v.get('mean')
        std_ = v.get('std')
        fq = v.get('frequency')
        mean_r = round(mean_, mean_round_decimals) if (mean_ is not None and isfinite(mean_)) else None
        std_r = round(std_,  mean_round_decimals) if (std_  is not None and isfinite(std_)) else None
        item = (k, mean_r)
        if keep_std:
            item += (std_r,)
        if keep_freq:
            item += (fq,)
        items.append(item)
    return tuple(items)


def _perf_equiv(v1, v2, rel_tol=0.02, abs_tol=1e-6):
    """Check if two perf vectors are equivalent under relative tolerance."""
    if len(v1) != len(v2):
        return False
    for a, b in zip(v1, v2):
        if not isfinite(a) or not isfinite(b):
            return False
        denom = max(abs(b), abs_tol)
        if abs(a - b) / denom > rel_tol:
            return False
    return True


def find_perf_equivalent_with_rel_tol(case_dict, change_dict,
                                      rel_tol=0.02,
                                      mean_round_decimals=6,
                                      min_occurrences=2):
    """
    Find elements that vary in mean but yield equivalent perf vectors
    (allowing small relative differences).
    Store also the full change_elements list for each timestep.
    """
    results = []

    for case_id, seeds in case_dict.items():
        for seed_id, series_list in seeds.items():
            arr = np.array(series_list, dtype=float)  # (T, 6)
            if arr.size == 0:
                continue

            good_rows = _finite_rows(arr)
            if good_rows.size < min_occurrences:
                continue

            # Raw change_elements lists (already stored in change_dict)
            ce_lists = change_dict[case_id][seed_id]

            # Candidate elements
            all_names = set()
            for t in good_rows:
                all_names.update(_to_map(ce_lists[t]).keys())

            for elem in sorted(all_names):
                buckets = defaultdict(list)  # (context_key, cluster_id) -> [(t, elem_mean, perf_vec, ce_list)]

                for t in good_rows:
                    ce_map = _to_map(ce_lists[t])
                    perf_vec = arr[t, :].tolist()
                    if any(x is None or not isfinite(x) for x in perf_vec):
                        continue

                    em = ce_map.get(elem, {}).get('mean')
                    if em is None or not isfinite(em):
                        continue

                    ctx_key = _context_key_without(elem, ce_map, mean_round_decimals)

                    # cluster assignment
                    assigned = False
                    for (ctx, cid), items in list(buckets.items()):
                        if ctx != ctx_key:
                            continue
                        rep_vec = items[0][2]
                        if _perf_equiv(perf_vec, rep_vec, rel_tol=rel_tol):
                            buckets[(ctx, cid)].append((int(t), float(em), perf_vec, ce_lists[t]))
                            assigned = True
                            break
                    if not assigned:
                        cid = len([1 for (ctx, _) in buckets.keys() if ctx == ctx_key])
                        buckets[(ctx_key, cid)].append((int(t), float(em), perf_vec, ce_lists[t]))

                # check equivalence in each cluster
                for (ctx_key, cid), items in buckets.items():
                    if len(items) < min_occurrences:
                        continue
                    means = sorted({ round(em, mean_round_decimals) for _, em, _, _ in items })
                    if len(means) >= 2:
                        ts = [t for t, _, _, _ in items]
                        rep_perf = items[0][2]
                        results.append({
                            "case_id": int(case_id),
                            "seed_id": int(seed_id),
                            "element": elem,
                            "context_key": list(ctx_key),
                            "perf_signature": rep_perf,          # representative perf vector
                            "distinct_means": means,
                            "timesteps": ts,
                            "count": len(items),
                            "change_elements": [ce for _, _, _, ce in items]  # store raw lists
                        })
    return results


def find_perf_equivalent_perf_only(case_dict, change_dict,
                                   rel_tol=0.02,
                                   min_occurrences=2):
    """
    Group timesteps whose performance vectors are equivalent (by _perf_equiv),
    ignoring which/how many elements changed. For each (case_id, seed_id),
    we cluster timesteps by a representative perf vector. Any cluster with
    size >= min_occurrences is reported.

    Returns: list of dicts with keys:
      - case_id, seed_id
      - perf_signature: representative perf vector for the cluster
      - timesteps: list[int] of matching timesteps
      - count: cluster size
      - change_elements: raw change_elements (one per timestep) for reference
    """
    results = []

    for case_id, seeds in case_dict.items():
        for seed_id, series_list in seeds.items():
            arr = np.array(series_list, dtype=float)  # shape: (T, D)
            if arr.size == 0:
                continue

            good_rows = _finite_rows(arr)  # user-provided helper: returns valid t indices
            if good_rows.size < min_occurrences:
                continue

            # change_dict lookup is optional; if missing, just store None
            ce_lists = change_dict.get(case_id, {}).get(seed_id, [])

            # ---- cluster timesteps by perf-equivalence only ----
            clusters = []  # list of {'rep': perf_vec, 'items': [(t, perf_vec, ce_list)]}

            for t in good_rows:
                perf_vec = arr[t, :].tolist()
                if any(x is None or not isfinite(x) for x in perf_vec):
                    continue

                assigned = False
                for c in clusters:
                    if _perf_equiv(perf_vec, c['rep'], rel_tol=rel_tol):
                        c['items'].append((int(t), perf_vec,
                                           ce_lists[t] if t < len(ce_lists) else None))
                        assigned = True
                        break
                if not assigned:
                    clusters.append({
                        'rep': perf_vec,
                        'items': [(int(t), perf_vec,
                                   ce_lists[t] if t < len(ce_lists) else None)]
                    })

            # ---- collect qualifying clusters ----
            for c in clusters:
                if len(c['items']) >= min_occurrences:
                    ts = [t for (t, _, _) in c['items']]
                    results.append({
                        "case_id": int(case_id),
                        "seed_id": int(seed_id),
                        "perf_signature": c['rep'],
                        "timesteps": ts,
                        "count": len(c['items']),
                        "change_elements": [ce for (_, _, ce) in c['items'] if ce is not None],
                    })

    return results


def find_perf_equivalents_global(case_dict,
                                 change_dict=None,
                                 rel_tol=0.02,
                                 abs_tol=1e-6,
                                 min_group_size=2,
                                 round_signature=4):
    """
    Pool ALL (case_id, seed_id, timestep) rows across case_dict, and group rows whose
    performance vectors are equivalent under _perf_equiv (ignoring which/how-many elements changed).

    Args
    ----
    case_dict: dict[case_id][seed_id] -> list[list[float]]
        Per (case, seed), a list over timesteps of performance vectors (same dimension D).
    change_dict: optional dict[case_id][seed_id] -> list[change_elements_at_t]
        If provided, we carry these along for provenance/debugging only.
    rel_tol, abs_tol: float
        Tolerances used by _perf_equiv.
    min_group_size: int
        Only emit groups with at least this many members (across any cases/seeds).
    round_signature: int or None
        If set, pre-bucket vectors by rounding each component to this many decimals.
        This greatly speeds up grouping; inside each bucket we still use _perf_equiv
        for exact membership.

    Returns
    -------
    groups: list[dict]
        Each dict has:
          - 'perf_signature': representative perf vector (list[float])
          - 'members': list of dicts with keys:
                {'case_id', 'seed_id', 'timestep', 'perf_vec', 'change_elements'}
          - 'count': number of members
    """
    # ---- 1) Flatten all rows with provenance ----
    rows = []  # list of (perf_vec, case_id, seed_id, t, change_elements)
    for case_id, seeds in case_dict.items():
        for seed_id, series_list in seeds.items():
            arr = np.array(series_list, dtype=float)  # shape (T, D)
            if arr.size == 0:
                continue
            # finite rows only
            finite_mask = np.all(np.isfinite(arr), axis=1)
            ce_list = (change_dict or {}).get(case_id, {}).get(seed_id, [])
            for t, ok in enumerate(finite_mask):
                if not ok:
                    continue
                perf_vec = arr[t, :].tolist()
                rows.append((
                    perf_vec,
                    int(case_id), int(seed_id), int(t),
                    ce_list[t] if t < len(ce_list) else None
                ))

    if not rows:
        return []

    # ---- 2) Optional pre-bucketing by rounded signature (for speed) ----
    def sig_key(vec):
        if round_signature is None:
            # single bucket: all together
            return ('_all',)
        return tuple(round(x, round_signature) for x in vec)

    buckets = defaultdict(list)
    for perf_vec, cid, sid, t, ce in rows:
        buckets[sig_key(perf_vec)].append((perf_vec, cid, sid, t, ce))

    # ---- 3) Within each bucket, build groups by _perf_equiv ----
    groups = []
    for bucket in buckets.values():
        representatives = []  # list of (rep_vec, members_list)
        for perf_vec, cid, sid, t, ce in bucket:
            placed = False
            for i, (rep_vec, members) in enumerate(representatives):
                if _perf_equiv(perf_vec, rep_vec, rel_tol=rel_tol, abs_tol=abs_tol):
                    members.append({
                        "case_id": cid, "seed_id": sid, "timestep": t,
                        "perf_vec": perf_vec, "change_elements": ce
                    })
                    placed = True
                    break
            if not placed:
                representatives.append((
                    perf_vec,
                    [{
                        "case_id": cid, "seed_id": sid, "timestep": t,
                        "perf_vec": perf_vec, "change_elements": ce
                    }]
                ))

        # collect groups meeting the size threshold
        for rep_vec, members in representatives:
            if len(members) >= min_group_size:
                groups.append({
                    "perf_signature": rep_vec,
                    "members": members,
                    "count": len(members),
                })

    # Optional: sort groups by size descending
    groups.sort(key=lambda g: g["count"], reverse=True)
    return groups


def group_spikes(spike_summary):
    grouped = OrderedDict()  # preserves insertion order
    for s in spike_summary:
        key = (s["case_id"], s["seed_id"], s["timestep"])
        if key not in grouped:
            grouped[key] = {
                "case_id": s["case_id"],
                "seed_id": s["seed_id"],
                "timestep": s["timestep"],
                # collect series and their values
                "series": [s["series"]],
                "values": [s["value"]],
                # keep prev/curr/next as-is
                "window": s["window"],
                "context": s["context"]
            }
        else:
            grouped[key]["series"].append(s["series"])
            grouped[key]["values"].append(s["value"])
            # prev/curr/next remain as-is

    # optional: sort series (and align values) for readability
    out = []
    for item in grouped.values():
        # sort series and reorder values accordingly
        pairs = sorted(zip(item["series"], item["values"]), key=lambda x: x[0])
        item["series"] = [p[0] for p in pairs]
        item["values"] = [p[1] for p in pairs]
        out.append(item)
    return out


def main():
    pkl_file_path = "stable_cases_train/qpme/simulation_results.pkl"
    _request_element_list = ["Workload", "Auth_login", "VerifyCode_verifyCode", "Admin_getAllUser", "User_getAllUser",
                             "Admin_updateAllUser", "User_updateAllUser", "VerifyCode_generateCode",
                             "Order_findAllOrder",
                             "Rebook_rebook", "Order_getOrder"]
    _real_url_type = ["/login", "/getAllUser", "/updateUser", "/rebook", "/generateCode", "/findAllOrder"]

    with open(pkl_file_path, 'rb') as f:
        loaded_data = pickle.load(f)

    case_dict = defaultdict(lambda: defaultdict(list))
    change_dict = defaultdict(lambda: defaultdict(list))

    index = 0
    saturation_elements = []
    for element in loaded_data:
        case_id = element['case_id']
        seed_id = element['seed_id']
        change_elements = element['change_elements']
        rank_list = element['mean_st_ranked_list']  # e.g. [(elem,score),...]
        rank_element_list = [e[1] for e in rank_list]  # only take element names
        if 100 > max(rank_element_list) > 50:
            saturation_elements.append(change_elements)

        # element -> rank index
        # rank_pos = {e: i for i, e in enumerate(rank_element_list)}
        # rank_id_list = [rank_pos[it] for it in _request_element_list if it in rank_pos]

        perf_list = []
        for real_url in _real_url_type:
            perf = parse_probs(f"./stable_cases_train/qpme/analysis_result_{index}", real_url)
            if perf > 300:
                perf = None
            perf_list.append(perf)
        # append this rank_id_list to the dict entry for this case_id
        case_dict[case_id][seed_id].append(perf_list)
        change_dict[case_id][seed_id].append(change_elements)
        index += 1

    with open("figures_train/saturation_summary.json", "w", encoding="utf-8") as f:
        json.dump(saturation_elements, f, ensure_ascii=False, indent=2)

    perf_equiv_rel = find_perf_equivalent_perf_only(
        case_dict, change_dict,
        rel_tol=0.05,  # allow relative difference
        min_occurrences=2
    )

    with open("figures_train/performance_equivalent_summary.json", "w", encoding="utf-8") as f:
        json.dump(perf_equiv_rel, f, ensure_ascii=False, indent=2)

    case_dict = {cid: dict(sdict) for cid, sdict in case_dict.items()}
    change_dict = {cid: dict(sdict) for cid, sdict in change_dict.items()}
    spike_summary = []

    for case_id, seeds in case_dict.items():
        for seed_id, series_list in seeds.items():
            arr = np.array(series_list, dtype=float)  # shape (T, D)
            timesteps = np.arange(arr.shape[0])

            plt.figure(figsize=(10, 6))
            for col in range(arr.shape[1]):
                plt.plot(timesteps, arr[:, col], label=f"series_{col}")
                ctx_window = 3
                # detect spikes for this series
                idxs, vals = detect_spikes_neighbors(arr[:, col], k=10.0, window=ctx_window)

                for t, v in zip(idxs, vals):
                    # build prev/curr/next maps safely
                    prev_ce = _to_map(change_dict[case_id][seed_id][t - 1]) if t - 1 >= 0 else {}
                    curr_ce = _to_map(change_dict[case_id][seed_id][t])
                    next_ce = _to_map(change_dict[case_id][seed_id][t + 1]) if t + 1 < len(
                        change_dict[case_id][seed_id]) else {}
                    changes_seq = change_dict[case_id][seed_id]  # list aligned with timesteps
                    T = len(changes_seq)
                    start = max(0, t - ctx_window)
                    end = min(T - 1, t + ctx_window)

                    # collect mapped change elements for [t-window, ..., t, ..., t+window]
                    context = []
                    for ti in range(start, end + 1):
                        context.append({
                            "timestep": int(ti),
                            "offset": int(ti - t),  # negative = before, positive = after
                            "changes": _to_map(changes_seq[ti])  # your existing mapping helper
                        })

                    spike_summary.append({
                        "case_id": int(case_id),
                        "seed_id": int(seed_id),
                        "series": int(col),
                        "timestep": int(t),
                        "value": float(v) if np.isfinite(v) else None,
                        "window": int(ctx_window),
                        "context": context  # full range t-window ... t+window
                    })

                    # summarize mean changes
                    changes = _diff_means(prev_ce, curr_ce, next_ce, tol=1e-9)
                    freq_changes = _maybe_freq_change(prev_ce, curr_ce, next_ce)

                    # pretty print
                    print(f"[SPIKE] case={case_id} seed={seed_id} series={col} t={t} value={v:.4f}")
                    if freq_changes:
                        for key, pf, cf, nf in freq_changes:
                            print(f"  {key}.frequency: prev={pf} -> curr={cf} -> next={nf}")

                    if not changes:
                        print("  (no mean changes vs neighbors)")
                    else:
                        # Sort by |Δprev| descending (use |Δnext| as tiebreaker)
                        def _abs_or_zero(x):
                            return abs(x) if (x is not None and isfinite(x)) else 0.0

                        changes.sort(key=lambda z: (_abs_or_zero(z[4]), _abs_or_zero(z[5])), reverse=True)

                        print(
                            "  element                              prev       curr       next       Δprev       Δnext")
                        print(
                            "  -------------------------------------------------------------------------------------------")
                        for e, pm, cm, nm, dprev, dnext in changes:
                            print(
                                f"  {e:<34} {_fmt_float(pm):>9}  {_fmt_float(cm):>9}  {_fmt_float(nm):>9}  {_fmt_float(dprev):>9}  {_fmt_float(dnext):>9}")

            plt.xlabel("Time step")
            plt.ylabel("Value")
            plt.title(f"Case {case_id}, Seed {seed_id}")
            plt.legend()
            plt.tight_layout()

            fname = f"figures_train/case_{case_id}_seed_{seed_id}.png"
            plt.savefig(fname)
            plt.close()

    grouped_spike = group_spikes(spike_summary)
    print(len(grouped_spike))
    print(len(saturation_elements))
    print(len(perf_equiv_rel))
    # save one JSON file with all spikes
    with open("figures_train/spike_summary.json", "w", encoding="utf-8") as f:
        json.dump(grouped_spike, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()