import ast
import pickle
import re
import numpy as np
from xgboost import XGBRegressor
from statistics import mean
from workload_file_modifier import TestCases
from single_pod_model import encode_config


def wall_times_in_line(line: str):
    """Return all wall_time floats found on this line via literal_eval, else regex."""
    vals = []
    try:
        obj = ast.literal_eval(line)
        if isinstance(obj, dict):
            if "wall_time" in obj:
                vals.append(float(obj["wall_time"]))
            ff = obj.get("ffmpeg")
            if isinstance(ff, dict) and "wall_time" in ff:
                vals.append(float(ff["wall_time"]))
            if vals:
                return vals
    except Exception:
        pass
    # fallback: regex
    return [float(m) for m in re.findall(
        r"""['"]wall_time['"]\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)""", line
    )]


def parse_log_to_Xy(log_path: str):
    # Headers marking a new selection block (optional; implicit rollover also supported)
    header_re      = re.compile(r"^\s*Active learning selection\s*$", re.IGNORECASE)

    # Setting lines for each family
    ffmpeg_set_re  = re.compile(r"^\s*-\s*Setting\s+.*?ffmpeg\.(codec|preset|crf)\s+to\s+(.+?)\s*$")
    lrzip_set_re   = re.compile(r"^\s*-\s*Setting\s+.*?lrzip\.(algorithm|level|window|nice|processor)\s+to\s+(.+?)\s*$")

    REQUIRED = {
        "ffmpeg": {"codec", "preset", "crf"},
        "lrzip":  {"algorithm", "level", "window", "nice", "processor"},
    }
    FIRST_KEY = {"ffmpeg": "codec", "lrzip": "algorithm"}  # used for implicit new-block detection

    X, y = [], []
    cfg = {}           # current key->value map
    family = None      # 'ffmpeg' or 'lrzip'
    walls = []         # list of float wall times (provide wall_times_in_line elsewhere)

    def _emit_if_ready():
        nonlocal cfg, walls, family, X, y
        if not family:
            return
        need = REQUIRED[family]
        if need <= cfg.keys():
            if family == "ffmpeg":
                crf_val = cfg.get("crf")
                try:
                    crf_val = int(str(crf_val))
                except Exception:
                    pass
                X.append([cfg["codec"], cfg["preset"], crf_val])
            else:  # lrzip
                def _as_int(v):
                    try:
                        return int(str(v))
                    except Exception:
                        return v
                X.append([
                    cfg["algorithm"],
                    _as_int(cfg["level"]),
                    _as_int(cfg["window"]),
                    _as_int(cfg["nice"]),
                    _as_int(cfg["processor"]),
                ])
            y.append(mean(walls) if walls else float("nan"))
        # reset
        cfg, walls, family = {}, [], None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.rstrip("\n")

            # Explicit block header → finalize current block
            if header_re.match(s):
                _emit_if_ready()
                continue

            # Try ffmpeg setting
            m = ffmpeg_set_re.match(s)
            if m:
                key, val = m.group(1), m.group(2).strip().strip('"').strip("'")
                # Switching families or starting a new complete block → emit previous
                if family and family != "ffmpeg":
                    _emit_if_ready()
                # Implicit new block when first key repeats after completion
                if family == "ffmpeg" and REQUIRED["ffmpeg"] <= cfg.keys() and key == FIRST_KEY["ffmpeg"]:
                    _emit_if_ready()
                family = "ffmpeg"
                cfg[key] = val
                continue

            # Try lrzip setting
            m = lrzip_set_re.match(s)
            if m:
                key, val = m.group(1), m.group(2).strip().strip('"').strip("'")
                if family and family != "lrzip":
                    _emit_if_ready()
                if family == "lrzip" and REQUIRED["lrzip"] <= cfg.keys() and key == FIRST_KEY["lrzip"]:
                    _emit_if_ready()
                family = "lrzip"
                cfg[key] = val
                continue

            # Measurements (collect all wall times found on this line)
            if "wall_time" in s:
                walls.extend(wall_times_in_line(s))  # assumes you already have this helper

    # Finalize at EOF
    _emit_if_ready()

    return X, y


# --- usage ---
if __name__ == "__main__":
    X_convert = []
    X, y = parse_log_to_Xy("../ori/single_pod_lrzip.0")
    print(f"Parsed {len(X)} blocks.")
    # work_model_address = "../muBench_changed_files/teastore/WorkModel.json"
    work_model_address = "../muBench_changed_files/trainticket_mix/WorkModel.json"
    tc = TestCases(work_model_address)
    tc.load_workload_model()
    tc.get_config_space("s9", "lrzip_test_only")
    conf_space = tc.config_space
    for i, (xrow, yval) in enumerate(zip(X, y), 1):
        print(f"{i:03d}: X={xrow} | y={yval:.9f}")
        '''config = {
            's4.internal_service.test_only.config_tester.ffmpeg.codec': xrow[0],
            's4.internal_service.test_only.config_tester.ffmpeg.preset': xrow[1],
            's4.internal_service.test_only.config_tester.ffmpeg.crf': int(xrow[2]),
        }'''
        config = {
            's9.internal_service.lrzip_test_only.config_tester.lrzip.algorithm': xrow[0],
            's9.internal_service.lrzip_test_only.config_tester.lrzip.level': int(xrow[1]),
            's9.internal_service.lrzip_test_only.config_tester.lrzip.window': int(xrow[2]),
            's9.internal_service.lrzip_test_only.config_tester.lrzip.nice': int(xrow[3]),
            's9.internal_service.lrzip_test_only.config_tester.lrzip.processor': int(xrow[4]),
        }
        X_convert.append(encode_config(config, conf_space))
    print(X_convert)
    print(y)

    length = 38
    save_file_name = f"../ori/lrzip_final_model_{length}.pkl"
    X_reduced = X_convert[0:length]
    y_reduced = y[0:length]
    final_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                               random_state=42)
    final_model.fit(np.array(X_reduced), np.array(y_reduced))
    with open(save_file_name, "wb") as f:
        pickle.dump(final_model, f)