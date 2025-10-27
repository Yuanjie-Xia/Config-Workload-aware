import random
import statistics
import time
import numpy as np
import pickle
import requests
import xml.etree.ElementTree as ET

from scipy.stats import spearmanr
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, train_test_split

from Deployer import Deployer
from workload_file_modifier import TestCases
from test_perf_spike import convert_mean2config
from single_pod_model import sample_config, encode_config, max_std_sampling, rank_evaluation_kfold, qpme_process

from modAL.models import ActiveLearner, CommitteeRegressor
from sklearn.utils import resample
from xgboost import XGBRegressor
from deepPerf import DeepPerfRegressor, mre


_request_element_list = ["Workload", "Auth_login", "VerifyCode_verifyCode", "Admin_getAllUser", "User_getAllUser",
                             "Admin_updateAllUser", "User_updateAllUser", "VerifyCode_generateCode",
                             "Order_findAllOrder",
                             "Rebook_rebook", "Order_getOrder"]

_simplize_request_element_list = ["Workload", "s5_url_1", "s6_url_1", "s7_url_2", "s8_url_2", "s7_url_3",
                                      "s8_url_3", "s6_url_4", "s9_url_5", "s10_url_6", "s9_url_6"]
_servers = ["s5", "s6", "s7", "s8", "s9", "s10"]
_url_type = ["url_1", "url_2", "url_3", "url_4", "url_5", "url_6"]
_url_type_start_node = ["s5", "s7", "s7", "s6", "s9", "s10"]
start_node_map = dict(zip(_url_type, _url_type_start_node))
_real_url_type = ["/login", "/getAllUser", "/updateUser", "/generateCode", "/findAllOrder", "/rebook"]
request_map = dict(zip(_request_element_list, _simplize_request_element_list))


def measure_request_time(url_type):
    try:
        headers = {"x-requesttype": f"{url_type}"}

        start = time.time()  # start timer
        resp = requests.get(
            f"http://129.97.165.134:31114/{start_node_map[url_type]}",
            headers=headers,
            timeout=100
        )
        resp.raise_for_status()
        end = time.time()  # end timer

        elapsed = end - start
        print(f"Request response time: {elapsed:.4f} seconds")
        return elapsed

    except Exception as e:
        print("Error in measure_request_time:", e)
        return float("inf")


def measure_perf_certain(url_type):
    try:
        headers = {
            "x-requesttype": f"{url_type}"
        }
        resp = requests.get(
            f"http://129.97.165.134:31114/{start_node_map[url_type]}",
            headers=headers,
            timeout=100,
        )
        resp.raise_for_status()
        data = resp.json()

        # Get top-level ffmpeg.wall_time only
        # print(data)
        wall_time = data.get("ffmpeg", {}).get("wall_time", None)
        print(wall_time)
        if wall_time is None:
            print("Wall_time not found in response!")
            print("Full response:", data)
            return float("inf")  # Penalize failed responses
        return wall_time
    except Exception as e:
        print("Error in measure_performance:", e)
        return float("inf")


def run_experiment_certain(testcase, deployer, config, url_type, repeat_times=10):
    for key in config:
        value = config[key]
        print(f"  - Setting {key} to {value}")
        testcase.modify_workload_model(key, value)
    deployer.create_workmodel_configmap_data()
    deployer.undeploy_configmap()
    deployer.deploy_configmap()
    deployer.restart_deployments()
    time.sleep(3)
    perf_list = []
    for i in range(repeat_times):
        perf = measure_request_time(url_type)
        perf_list.append(perf)

    # Remove the max and min values
    perf_list.remove(max(perf_list))
    perf_list.remove(min(perf_list))

    # Return the average of the remaining values
    avg_perf = sum(perf_list) / len(perf_list)
    std_perf = np.std(perf_list, ddof=1)
    print(f"Standard Deviation: {std_perf}")
    return avg_perf


def active_learning_committee(test_case, deployer, url, rounds=100, n_committee=5):

    X, y = [], []
    seen_configs = []
    conf_space = test_case.config_space

    # Step 1: Initialize with random points
    init_configs = []
    while len(init_configs) < n_committee:
        config = sample_config(conf_space)
        if config not in init_configs:
            init_configs.append(config)
            seen_configs.append(config)

    for config in init_configs:
        perf = run_experiment_certain(test_case, deployer, config, url)
        encoded_config = encode_config(config, conf_space)
        X.append(encoded_config)
        y.append(perf)

    # Step 2: Bootstrap training sets for each learner
    learner_list = []
    for _ in range(n_committee):
        X_bootstrap, y_bootstrap = resample(X, y)
        learner = ActiveLearner(
            estimator=XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42),
            query_strategy=max_std_sampling,
            X_training=np.array(X_bootstrap),
            y_training=np.array(y_bootstrap)
        )
        learner_list.append(learner)

    # Step 3: Initialize committee
    committee = CommitteeRegressor(
        learner_list=learner_list,
        query_strategy=max_std_sampling
    )

    # Step 4: Active learning loop
    for i in range(len(init_configs), rounds):
        # Generate candidate pool
        candidates = [sample_config(conf_space) for _ in range(100)]
        candidates = [c for c in candidates if c not in seen_configs]
        if not candidates:
            print("No new configs left.")
            break

        X_pool = [encode_config(c, conf_space) for c in candidates]
        # Step 5: Query next config
        # X_pool_np = list(map(list, X_pool))
        query_idx, _ = committee.query(X_pool)
        config = candidates[int(query_idx[0])]
        print(config)

        # Step 6: Evaluate and teach
        print("-------------------------------------------------------")

        perf = run_experiment_certain(test_case, deployer, config, url)
        print(f"Round {i + 1}: Config={config}, Performance={perf:.4f}")

        start_time = time.time()
        X_new = np.array([encode_config(config, conf_space)])
        y_new = np.array([perf])
        committee.teach(X_new, y_new)
        X.append(X_new[0])
        y.append(y_new[0])
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time}")

        if i > 10:
            print(f"length of X is {len(X)}")
            X_eval = np.array(X)
            y_eval = np.array(y)
            rank_evaluation_kfold(X_eval, y_eval, top_k=5)

            # Train a single model on the training split
            eval_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)

            # 10-fold cross-validation
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            r2_scores = cross_val_score(eval_model, X_eval, y_eval, cv=kf, scoring='r2')
            y_pred = cross_val_predict(eval_model, X_eval, y_eval, cv=kf)

            # Relative error
            relative_errors = np.abs((y_eval - y_pred) / (y_eval + 1e-8))
            mean_rel_error = np.mean(relative_errors)

            # Save data
            save_data = {
                "X": X,
                "y": y,
                "model": eval_model
            }
            with open(f"modAL_all_{url}.pkl", "wb") as f:
                pickle.dump(save_data, f)

            # Output results
            print(
                f"[10-Fold CV] Mean R²: {np.mean(r2_scores):.4f}, Std R²: {np.std(r2_scores):.4f}, Mean Relative Error: {mean_rel_error:.4f}")

        # Stop criteria
            #if r2 > 0.95 or mean_rel_error < 0.05:
            #    print("Stopping early: performance threshold met.")
            #    break

        seen_configs.append(config)
    start_time = time.time()
    final_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)
    final_model.fit(np.array(X), np.array(y))
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time}")
    return final_model


def deepperf_style_fine_tune(
    X, y,
    *,
    base_params: dict | None = None,
    layer_range = range(2, 15),                                   # 2..14 layers
    lr_range = None,                                              # logspace(1e-4..1e-1)
    lambda_range = None,                                          # 1e-2 .. 1e3
    val_size: float = 0.25,
    random_state: int = 42,
    epochs_tune: int = 80,                                        # shorter than full training
    patience_tune: int = 10,
    batch_size: int = 128,
    device: str | None = None,
    verbose: int = 1
):
    """
    Returns a dict of best hyperparameters: {'n_layers', 'learning_rate', 'l1'}
    following the two-stage search in DeepPerf (layers+lr, then lambda/L1).
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    if lr_range is None:
        lr_range = np.logspace(np.log10(1e-4), np.log10(1e-1), 4)
    if lambda_range is None:
        lambda_range = np.logspace(-2, 3, 30)   # 1e-2..1e3

    # split "training1" and "training2" (the validation in the paper)
    X_tr1, X_tr2, y_tr1, y_tr2 = train_test_split(
        X, y, test_size=val_size, random_state=random_state
    )

    # base knobs we do NOT tune here (keep your committee’s choices)
    defaults = {
        "hidden_units": 128,
        "dropout": 0.10,
        "activation": "relu",
        "normalize_y": False,
        "epochs": epochs_tune,
        "patience": patience_tune,
        "batch_size": batch_size,
        "random_state": random_state,
        "device": device
    }
    if base_params:
        defaults.update({k: v for k, v in base_params.items()
                         if k in {"hidden_units","dropout","activation","normalize_y",
                                  "epochs","patience","batch_size","random_state","device"}})

    # --- Stage 1: search (n_layers × learning_rate) ---
    abs_err_train = np.full((max(layer_range)+1, len(lr_range)), np.inf, dtype=np.float32)
    abs_err_val   = np.full((max(layer_range)+1, len(lr_range)), np.inf, dtype=np.float32)
    best_val_per_layer = []   # [(val_err_best_for_layer, best_lr_for_layer, n_layer)]

    early_worse_count = 0
    best_so_far = np.inf

    for n_layer in layer_range:
        # evaluate all LR for this #layers
        for lr_idx, lr in enumerate(lr_range):
            est = DeepPerfRegressor(
                n_layers=n_layer, learning_rate=float(lr), l1=defaults.get("l1", 1e-3),
                hidden_units=defaults["hidden_units"], dropout=defaults["dropout"],
                activation=defaults["activation"], normalize_y=defaults["normalize_y"],
                epochs=defaults["epochs"], patience=defaults["patience"],
                batch_size=defaults["batch_size"], random_state=defaults["random_state"],
                device=defaults["device"]
            )
            est.fit(X_tr1, y_tr1)

            # train abs error
            yhat_tr1 = est.predict(X_tr1)
            tr_abs = float(np.mean(np.abs(yhat_tr1 - y_tr1)))
            abs_err_train[int(n_layer), lr_idx] = tr_abs

            # validation abs error
            yhat_tr2 = est.predict(X_tr2)
            val_abs = float(np.mean(np.abs(yhat_tr2 - y_tr2)))
            abs_err_val[int(n_layer), lr_idx] = val_abs

        # choose lr with smallest train error (like the paper)
        lr_best_idx = int(np.argmin(abs_err_train[int(n_layer), :]))
        lr_best = float(lr_range[lr_best_idx])
        val_best = float(abs_err_val[int(n_layer), lr_best_idx])
        best_val_per_layer.append((val_best, lr_best, n_layer))

        # early layer stopping heuristic (if getting worse 2 times in a row, stop)
        if val_best + 1e-9 < best_so_far:
            best_so_far = val_best
            early_worse_count = 0
        else:
            early_worse_count += 1
        if early_worse_count >= 2:
            if verbose:
                print("[fine-tune] early stop over layers (no improvement for 2 layers).")
            break

    # pick the layer with min validation error
    if not best_val_per_layer:
        # fallback
        n_layer_opt, lr_opt = 3, float(lr_range[0])
    else:
        best_layer_tuple = min(best_val_per_layer, key=lambda t: t[0])
        _, lr_opt, n_layer_opt = best_layer_tuple

    # refine LR at fixed n_layer_opt (evaluate by val abs error)
    val_errs_for_lr = []
    for lr_idx, lr in enumerate(lr_range):
        est = DeepPerfRegressor(
            n_layers=n_layer_opt, learning_rate=float(lr), l1=defaults.get("l1", 1e-3),
            hidden_units=defaults["hidden_units"], dropout=defaults["dropout"],
            activation=defaults["activation"], normalize_y=defaults["normalize_y"],
            epochs=defaults["epochs"], patience=defaults["patience"],
            batch_size=defaults["batch_size"], random_state=defaults["random_state"],
            device=defaults["device"]
        )
        est.fit(X_tr1, y_tr1)
        yhat_tr2 = est.predict(X_tr2)
        val_abs = float(np.mean(np.abs(yhat_tr2 - y_tr2)))
        val_errs_for_lr.append(val_abs)
    lr_opt = float(lr_range[int(np.argmin(val_errs_for_lr))])

    if verbose:
        print(f"[fine-tune] best n_layers={n_layer_opt}, best lr={lr_opt:g}")

    # --- Stage 2: search lambda (L1) ---
    best_lambda = None
    best_val_abs = np.inf
    for lambd in lambda_range:
        est = DeepPerfRegressor(
            n_layers=n_layer_opt, learning_rate=lr_opt, l1=float(lambd),
            hidden_units=defaults["hidden_units"], dropout=defaults["dropout"],
            activation=defaults["activation"], normalize_y=defaults["normalize_y"],
            epochs=defaults["epochs"], patience=defaults["patience"],
            batch_size=defaults["batch_size"], random_state=defaults["random_state"],
            device=defaults["device"]
        )
        est.fit(X_tr1, y_tr1)
        yhat_tr2 = est.predict(X_tr2)
        val_abs = float(np.mean(np.abs(yhat_tr2 - y_tr2)))
        if val_abs < best_val_abs:
            best_val_abs = val_abs
            best_lambda = float(lambd)

    if verbose:
        print(f"[fine-tune] best l1={best_lambda:g} (val_abs={best_val_abs:.6f})")

    return {
        "n_layers": int(n_layer_opt),
        "learning_rate": float(lr_opt),
        "l1": float(best_lambda),
    }


def active_learning_committee_dp(test_case, deployer, url, rounds=100, n_committee=5, fine_tune_every=3, verbose_tune=1):
    X, y = [], []
    seen_configs = []
    conf_space = test_case.config_space

    # --- Step 1: seed with random unique configs ---
    init_configs = []
    while len(init_configs) < n_committee:
        config = sample_config(conf_space)
        if config not in init_configs:
            init_configs.append(config)
            seen_configs.append(config)

    for config in init_configs:
        perf = run_experiment_certain(test_case, deployer, config, url)
        X.append(encode_config(config, conf_space))
        y.append(perf)

    # --- Step 2: bootstrap a committee of DeepPerf learners ---
    learner_list = []
    for _ in range(n_committee):
        X_boot, y_boot = resample(X, y)
        learner = ActiveLearner(
            estimator=DeepPerfRegressor(
                n_layers=4,
                hidden_units=128,
                l1=1e-3,
                learning_rate=1e-3,
                dropout=0.10,
                epochs=200,          # tune if needed
                patience=20,
                batch_size=128,
                normalize_y=False,
                random_state=42,
                device=None          # auto: cuda if available else cpu
            ),
            query_strategy=max_std_sampling,
            X_training=np.array(X_boot, dtype=np.float32),
            y_training=np.array(y_boot, dtype=np.float32)
        )
        learner_list.append(learner)

    # --- Step 3: committee ---
    committee = CommitteeRegressor(
        learner_list=learner_list,
        query_strategy=max_std_sampling
    )

    # --- Step 4: active learning loop ---
    for i in range(len(init_configs), rounds):
        # candidate pool (skip duplicates)
        candidates = [sample_config(conf_space) for _ in range(100)]
        candidates = [c for c in candidates if c not in seen_configs]
        if not candidates:
            print("No new configs left.")
            break

        X_pool = [encode_config(c, conf_space) for c in candidates]

        # Step 5: query by committee std (disagreement)
        query_idx, _ = committee.query(X_pool)
        config = candidates[int(query_idx[0])]
        print(config)
        print("-------------------------------------------------------")

        # Step 6: run experiment and teach the committee
        perf = run_experiment_certain(test_case, deployer, config, url)
        print(f"Round {i + 1}: Config={config}, Performance={perf:.4f}")

        start_time = time.time()
        X_new = np.array([encode_config(config, conf_space)], dtype=np.float32)
        y_new = np.array([perf], dtype=np.float32)
        committee.teach(X_new, y_new)
        X.append(X_new[0])
        y.append(y_new[0])

        # ---- DeepPerf-style fine-tune every N steps after seeding ----
        steps_after_seed = i - len(init_configs) + 1
        if steps_after_seed > 0 and steps_after_seed % fine_tune_every == 0:
            print(f"[Fine-tune] After {steps_after_seed} acquired configs, tuning (layers, lr, l1)...")
            # Use the first learner's params as base (hidden_units/dropout/etc.)
            base_params = committee.learner_list[0].estimator.get_params()
            best = deepperf_style_fine_tune(
                X, y,
                base_params=base_params,
                val_size=0.25,
                random_state=42,
                epochs_tune=80, patience_tune=10, batch_size=128,
                device=base_params.get("device", None),
                verbose=verbose_tune
            )
            # Apply the best hyperparams to every learner and refit on ALL data
            X_all = np.asarray(X, dtype=np.float32)
            y_all = np.asarray(y, dtype=np.float32)
            for idx, learner in enumerate(committee.learner_list, start=1):
                est: DeepPerfRegressor = learner.estimator
                est.set_params(**best)
                est.fit(X_all, y_all)  # refit
                # keep modAL bookkeeping consistent
                learner.X_training = X_all
                learner.y_training = y_all
                if verbose_tune:
                    print(f"[Fine-tune] Learner {idx} set to {best}")
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time}")
        if i > 10:
            print(f"length of X is {len(X)}")
            X_eval = np.array(X, dtype=np.float32)
            y_eval = np.array(y, dtype=np.float32)

            # your ranking eval
            rank_evaluation_kfold(X_eval, y_eval, top_k=5)

            # a single DeepPerf model for CV assessment
            eval_model = DeepPerfRegressor(
                n_layers=4, hidden_units=128, l1=1e-3,
                learning_rate=1e-3, dropout=0.10,
                epochs=200, patience=20, batch_size=128,
                normalize_y=False, random_state=42
            )
            eval_model.set_params(**best)

            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            r2_scores = cross_val_score(eval_model, X_eval, y_eval, cv=kf, scoring='r2')
            y_pred = cross_val_predict(eval_model, X_eval, y_eval, cv=kf)

            # Mean Relative Error (safer with eps)
            mean_rel_error = mre(y_eval, y_pred, eps=1e-12)

            # Save data (store model weights separately to avoid giant pickles)
            eval_model.fit(X_eval, y_eval)
            model_path = f"deepperf_cv_{url}.pt"
            eval_model.save(model_path)

            save_data = {
                "X": X,
                "y": y,
                "model_path": model_path,
                "model_params": eval_model.get_params()
            }
            with open(f"modAL_all_{url}.pkl", "wb") as f:
                pickle.dump(save_data, f)

            print(f"[10-Fold CV] Mean R²: {np.mean(r2_scores):.4f}, "
                  f"Std R²: {np.std(r2_scores):.4f}, "
                  f"Mean Relative Error: {mean_rel_error:.4f}")

        seen_configs.append(config)

    # --- Final model trained on all collected data ---
    start_time = time.time()
    final_model = DeepPerfRegressor(
        n_layers=4, hidden_units=128, l1=1e-3,
        learning_rate=1e-3, dropout=0.10,
        epochs=300, patience=30, batch_size=128,
        normalize_y=False, random_state=42
    )
    # Optional: a short final tune pass
    best_final = deepperf_style_fine_tune(X, y, base_params=final_model.get_params(),
                                          epochs_tune=100, patience_tune=12, verbose=0)
    final_model.set_params(**best_final)
    final_model.fit(np.array(X, dtype=np.float32), np.array(y, dtype=np.float32))
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time}")
    return final_model


def active_learning_model():
    for url in _url_type:
        work_model_address = "./muBench_changed_files/trainticket_mix/WorkModel.json"
        tc = TestCases(work_model_address)
        tc.load_workload_model()
        tc.get_config_space(target_pod=None, internal_service=url)
        # tc.get_config_space()

        import re
        space = tc.config_space
        pat = re.compile(r"\.internal_service\.(ffmpeg_test_only|lrzip_test_only)\.", re.IGNORECASE)
        tc.config_space = {k: v for k, v in space.items() if not pat.search(k)}

        random.seed(42)
        deployer = Deployer(['s5', 's6', 's7', 's8', 's9', 's10'], tc.workload_json, namespace="train")
        # deployer.update_dns_setting()
        time.sleep(3)
        deployer.create_workmodel_configmap_data()
        deployer.undeploy_configmap()
        deployer.deploy_configmap()
        time.sleep(10)
        print("Active learning selection")
        _final_model = active_learning_committee(tc, deployer, url, rounds=240)
        with open(f"final_model_{url}.pkl", "wb") as f:
            pickle.dump(_final_model, f)
        print("Model and config space saved to configspace.pkl")


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
                            print(f"[QPME] {element_name}: meanST = {meanST:.6f}")
    except Exception as e:
        print(f"[ERROR] Failed to parse analysis file: {e}")
        return []
    return meanST


def relative_errors(true, pred):
    return [abs(p - t) / abs(t) if t != 0 else float('inf')
            for t, p in zip(true, pred)]


def active_learning_w_qpme():
    k = 0
    for url in _url_type:
        print("==========================")
        work_model_address = "./muBench_changed_files/trainticket_mix/WorkModel.json"
        tc = TestCases(work_model_address)
        tc.load_workload_model()
        tc.get_config_space()
        tc_1 = TestCases(work_model_address)
        tc_1.load_workload_model()
        tc_1.get_config_space("s6", "url_4")
        random.seed(42)
        deployer = Deployer(['s5', 's6', 's7', 's8', 's9', 's10'], tc.workload_json, namespace="train")
        # deployer.update_dns_setting()
        time.sleep(3)
        deployer.create_workmodel_configmap_data()
        deployer.undeploy_configmap()
        deployer.deploy_configmap()
        time.sleep(10)
        perf_list = []
        estimate_list = []
        save_data = []
        for index in range(0, 28):
            real_index = k * 28 + index
            print("------------------------------")
            _location_dict = {
                "ffmpeg": "./ori/final_model_28.pkl",
                "lrzip": "./ori/lrzip_final_model_28.pkl"
            }
            change_elements, ranked_list, mean_st_ranked_list = qpme_process(
                _location_dict, tc.config_space, real_index, "train", mix_status=True)
            url_index = _url_type.index(url)
            real_url = _real_url_type[url_index]
            probs_mean_st = parse_probs(f"./qpme/analysis_result_{real_index}", real_url)
            print(f"QPME estimation: {real_url} - {probs_mean_st}")
            estimate_list.append(probs_mean_st)

            _config_set = {}
            for elem in change_elements:
                if 'config' in elem:
                    _config_set.update(elem['config'])

            perf = run_experiment_certain(tc, deployer, _config_set, url)
            print(f"Real_perf: {perf}")
            perf_list.append(perf)
            if index > 10:
                error = relative_errors(perf_list, estimate_list)
                spearman_corr, _ = spearmanr(perf_list, estimate_list)
                print(f"Relative_error: {error}")
                print(f"Mean Relative error: {statistics.mean(error)}")
                print(f"Spearman correlation: {spearman_corr}")

            results = {
                "Estimate": probs_mean_st,
                "Real": perf,
                "change_elements": change_elements
            }
            save_data.append(results)
        with open(f"method_results_{url}.pkl", "wb") as f:
            pickle.dump(save_data, f)
        k = k + 1


if __name__ == '__main__':
    active_learning_w_qpme()