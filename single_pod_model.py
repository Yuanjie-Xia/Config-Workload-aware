import pickle
import random
import time

import numpy as np
import requests
from modAL.models import ActiveLearner, CommitteeRegressor
from modAL.utils.data import modALinput
from modAL.utils.selection import multi_argmax, shuffled_argmax
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.utils import resample
from xgboost import XGBRegressor

from Deployer import Deployer
from qpme_exec import qpme_running
from workload_file_modifier import TestCases


def regression_uncertainty_bootstrap(X_train, y_train, X_pool, n_models=10):
    preds_list = []

    for _ in range(n_models):
        # Bootstrap resample
        X_sample, y_sample = resample(X_train, y_train)

        # Train a new XGBRegressor
        model = XGBRegressor(n_estimators=100, verbosity=0)
        model.fit(X_sample, y_sample)

        # Predict on pool
        preds = model.predict(X_pool)
        preds_list.append(preds)

    preds_array = np.array(preds_list)  # Shape: (n_models, n_candidates)
    uncertainties = np.std(preds_array, axis=0)
    return uncertainties


def sample_config(config_space):
    return {k: random.choice(v) for k, v in config_space.items()}


def encode_config(config, config_space):
    encoded = []
    for key in config_space:
        val = config[key]
        options = config_space[key]

        if key.endswith('crf'):
            # Normalize crf to [0, 1]
            crf_min, crf_max = min(options), max(options)
            norm_val = (val - crf_min) / (crf_max - crf_min)
            encoded.append(norm_val)
        else:
            # One-hot encode categorical variables
            one_hot = [1 if val == opt else 0 for opt in options]
            encoded.extend(one_hot)
    return encoded


def measure_perf():
    try:
        headers = {
            "x-requesttype": "lrzip_test_only"
        }
        resp = requests.get(
            "http://129.97.165.134:31114/s9",
            headers=headers,
            timeout=100,
        )
        resp.raise_for_status()
        data = resp.json()

        # Get top-level ffmpeg.wall_time only
        print(data)
        # wall_time = data.get("ffmpeg", {}).get("wall_time", None)
        wall_time = data.get("lrzip", {}).get("wall_time", None)
        if wall_time is None:
            print("Wall_time not found in response!")
            print("Full response:", data)
            return float("inf")  # Penalize failed responses
        return wall_time
    except Exception as e:
        print("Error in measure_performance:", e)
        return float("inf")


def run_experiment(testcase, deployer, config, repeat_times=10):
    for key in config:
        value = config[key]
        print(f"  - Setting {key} to {value}")
        testcase.modify_workload_model(key, value)
    deployer.create_workmodel_configmap_data()
    deployer.undeploy_configmap()
    deployer.deploy_configmap()
    deployer.restart_deployments()
    time.sleep(30)
    perf_list = []
    for i in range(repeat_times):
        perf = measure_perf()
        perf_list.append(perf)

    # Remove the max and min values
    perf_list.remove(max(perf_list))
    perf_list.remove(min(perf_list))

    # Return the average of the remaining values
    avg_perf = sum(perf_list) / len(perf_list)
    std_perf = np.std(perf_list, ddof=1)
    print(f"Standard Deviation: {std_perf}")
    return avg_perf


def max_std_sampling(regressor: BaseEstimator, X: modALinput,
                     n_instances: int = 1, random_tie_break=False,
                     **predict_kwargs) -> np.ndarray:
    """
    Regressor standard deviation sampling strategy.
    """
    _, std = regressor.predict(X, return_std=True, **predict_kwargs)
    std = np.array(std).reshape(len(X),)

    if not random_tie_break:
        print(multi_argmax(std, n_instances=n_instances))
        return multi_argmax(std, n_instances=n_instances)
    else:
        print(shuffled_argmax(std, n_instances=n_instances))
        return shuffled_argmax(std, n_instances=n_instances)


def rank_evaluation_kfold(X, y, model=None, n_splits=10, top_k=5):
    if model is None:
        model = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )

    spearman_scores = []
    precision_at_k_scores = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = np.array(X)[train_idx], np.array(X)[test_idx]
        y_train, y_test = np.array(y)[train_idx], np.array(y)[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Spearman's Rank Correlation
        spearman_corr, _ = spearmanr(y_test, y_pred)
        spearman_scores.append(spearman_corr)

        # Precision@K
        true_top_k = np.argsort(y_test)[-top_k:]
        pred_top_k = np.argsort(y_pred)[-top_k:]
        overlap = len(set(true_top_k).intersection(set(pred_top_k)))
        precision_at_k = overlap / top_k
        precision_at_k_scores.append(precision_at_k)

    # Report average metrics
    avg_spearman = np.mean(spearman_scores)
    avg_precision_k = np.mean(precision_at_k_scores)

    print(f"[10-Fold Rank Eval] Avg Spearman's Rank Correlation: {avg_spearman:.4f}")
    print(f"[10-Fold Rank Eval] Avg Precision@{top_k}: {avg_precision_k:.4f}")
    return avg_spearman, avg_precision_k


def active_learning_committee(test_case, deployer, rounds=100, n_committee=5):

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
        perf = run_experiment(test_case, deployer, config)
        X.append(encode_config(config, conf_space))
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

        perf = run_experiment(test_case, deployer, config)
        print(f"Round {i + 1}: Config={config}, Performance={perf:.4f}")

        X_new = np.array([encode_config(config, conf_space)])
        y_new = np.array([perf])
        committee.teach(X_new, y_new)
        X.append(X_new[0])
        y.append(y_new[0])

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
            with open("modAL_all.pkl", "wb") as f:
                pickle.dump(save_data, f)

            # Output results
            print(
                f"[10-Fold CV] Mean R²: {np.mean(r2_scores):.4f}, Std R²: {np.std(r2_scores):.4f}, Mean Relative Error: {mean_rel_error:.4f}")

        # Stop criteria
            #if r2 > 0.95 or mean_rel_error < 0.05:
            #    print("Stopping early: performance threshold met.")
            #    break

        seen_configs.append(config)
    final_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)
    final_model.fit(np.array(X), np.array(y))
    return final_model


def random_selection_baseline(test_case, deployer, rounds=100):
    X, y = [], []
    seen_configs = set()
    conf_space = test_case.config_space

    for i in range(rounds):
        # Sample a new config not yet seen
        attempts = 0
        while True:
            config = sample_config(conf_space)
            config_key = tuple(sorted(config.items()))
            if config_key not in seen_configs:
                seen_configs.add(config_key)
                break
            attempts += 1
            if attempts > 100:
                print("No new configs left.")
                return

        # Evaluate config
        perf = run_experiment(test_case, deployer, config)
        X.append(encode_config(config, conf_space))
        y.append(perf)
        print(f"[Random] Round {i+1}: Config={config}, Performance={perf:.4f}")

        # Evaluation (after enough data)
        if i >= 10:
            model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)
            X_np = np.array(X)
            y_np = np.array(y)
            rank_evaluation_kfold(X_np, y_np, top_k=5)
            # 10-fold CV: compute R² scores and cross-validated predictions
            r2_scores = cross_val_score(model, X_np, y_np, cv=10, scoring='r2')
            y_pred = cross_val_predict(model, X_np, y_np, cv=10)

            # Calculate mean relative error
            relative_errors = np.abs((y_np - y_pred) / (y_np + 1e-8))
            mean_rel = np.mean(relative_errors)

            # Save model and data
            model.fit(X_np, y_np)  # Final model fit on all data
            save_data = {
                "X": X,
                "y": y,
                "model": model
            }
            with open("random_baseline_all.pkl", "wb") as f:
                pickle.dump(save_data, f)

            print(f"[Random Eval] R² (mean over folds): {np.mean(r2_scores):.4f}, Mean Relative Error: {mean_rel:.4f}")


def reshape_config_space(config, config_space, element_i, reshaped_elements):
    _reshaped_pod_url = reshaped_elements[element_i].split('/')
    _reshaped_pod = _reshaped_pod_url[0]
    _reshaped_url = _reshaped_pod_url[1]
    _filtered_config_space = {
        k: v for k, v in config_space.items()
        if _reshaped_pod in k and _reshaped_url in k
    }
    _filtered_config = {
        k: v for k, v in config.items()
        if _reshaped_pod in k and _reshaped_url in k
    }
    return _filtered_config, _filtered_config_space


def generate_change_element_set_from_model(model_dict, conf_space, elements, simplized_elements, subject_type_list=None):
    """
    Use the model to predict mean performance for random configurations, mapped to elements.

    :param model: Trained regression model
    :param conf_space: Configuration space from test_case.config_space
    :param elements: List of strings (element names like "P/search", "Workload", etc.)
    :param std_range: Tuple for standard deviation range
    :return: List of dicts with keys: element, mean, std
    """
    change_list = []
    config = sample_config(conf_space)
    for i, element in enumerate(elements):
        if element == "Workload":
            # pred_mean = random.choice([0.8, 12, 15, 20, 30, 60, 100])
            # pred_mean = 4
            # frequency = random.choice(list(range(10, 76, 10))) # frequency per minute
            frequency = 6
            change_list.append({
                "element": element,
                "frequency": f"{frequency}",
                # "mean": f"{pred_mean}",
                "mean": f"{(60/frequency):.4f}",
                # "std": f"{(pred_mean/100):.4f}"
                "std": f"{(6/frequency):4f}"
            })
        else:
            if subject_type_list is not None:
                _subject_type = subject_type_list[element]
                model_selected = model_dict[_subject_type]
            else:
                _subject_type = None
                model_selected = next(iter(model_dict.values()))
            _filted_config, _filter_config_space = reshape_config_space(config, conf_space, i, simplized_elements)
            X_encoded = np.array([encode_config(_filted_config, _filter_config_space)])
            pred_mean = float(model_selected.predict(X_encoded)[0])
            print(f"Towards {_subject_type}, pred_mean is {pred_mean}")
            pred_std = f"{(pred_mean/100):.4f}"
            change_list.append({
                "element": element,
                "mean": f"{pred_mean:.4f}",
                "std": str(pred_std),
                "config": _filted_config,
            })

    return change_list


def generate_model():
    # work_model_address = "./muBench_changed_files/gateway_offloading/WorkModel.json"
    work_model_address = "./muBench_changed_files/trainticket_mix/WorkModel.json"
    tc = TestCases(work_model_address)
    tc.load_workload_model()
    # tc.get_config_space("s4", "test_only")
    tc.get_config_space("s9", "lrzip_test_only")
    random.seed(42)
    deployer = Deployer(['s5', 's6', 's7', 's8', 's9', 's10'], tc.workload_json, namespace="train")
    # deployer.update_dns_setting()
    time.sleep(3)
    deployer.create_workmodel_configmap_data()
    deployer.undeploy_configmap()
    deployer.deploy_configmap()
    time.sleep(10)
    # Active learning loop
    # print("Random selection: ")
    # random_selection_baseline(tc, deployer)
    print("Active learning selection")
    _final_model = active_learning_committee(tc, deployer)
    with open("ori/lrzip_final_model.pkl", "wb") as f:
        pickle.dump(_final_model, f)
    print("Model and config space saved to configspace.pkl")


def qpme_process(_model_file_path_list, _config_space, index, type="teastore", mix_status=False):
    _model_dict = {}
    for name, path in _model_file_path_list.items():
        # Load XGBoost model from .pkl
        with open(path, "rb") as f:
            _final_model = pickle.load(f)
        _model_dict[name] = _final_model
    _request_element_list = []
    _simplize_request_element_list = []
    _qpme_file = ""
    if type == "teastore":
        _request_element_list = ["Workload", "W/index", "W/login", "W/loginAction", "W/category", "W/product", "W/logout",
                             "P/categories_index", "P/categories_login", "A/login_loginAction", "P/category_category",
                             "P/getProduct_product", "A/logout_logout", "A/isLoggedIn_index", "I/icon_login",
                             "P/getUser_loginAction", "I/productImages_category", "R/getAds_product",
                             "P/category_logout", "I/icon_index", "A/isLoggedIn_login", "I/icon_loginAction",
                             "A/isLoggedIn_category", "I/product_product", "I/icon_logout", "A/isLoggedIn_product"]
        _simplize_request_element_list = ["Workload", "s0/url_1", "s0/url_2", "s0/url_3", "s0/url_4", "s0/url_5",
                                      "s0/url_6", "s1/url_1", "s1/url_2", "s2/url_3", "s1/url_4", "s1/url_5",
                                      "s2/url_6", "s2/url_1", "s3/url_2", "s1/url_3", "s3/url_4", "s4/url_5",
                                      "s1/url_6", "s3/url_1", "s2/url_2", "s3/url_3", "s2/url_4", "s3/url_5",
                                      "s3/url_6", "s2/url_5"]
        _qpme_file = "./qpme/teastore_model.qpe"
    if type == "train":
        _request_element_list = ["Workload", "Auth_login", "VerifyCode_verifyCode", "Admin_getAllUser",
                                 "User_getAllUser",
                                 "Admin_updateAllUser", "User_updateAllUser", "VerifyCode_generateCode",
                                 "Order_findAllOrder",
                                 "Rebook_rebook", "Order_getOrder"]

        _simplize_request_element_list = ["Workload", "s5/url_1", "s6/url_1", "s7/url_2", "s8/url_2", "s7/url_3",
                                          "s8/url_3", "s6/url_4", "s9/url_5", "s10/url_6", "s9/url_6"]
        _qpme_file = "./qpme/trainticket_model.qpe"

    _subject_type = None
    _request_subject_map = None
    if mix_status:
        if type == "teastore":
            _subject_type = ["Workload", "ffmpeg", "lrzip", "ffmpeg", "ffmpeg", "ffmpeg", "lrzip", "lrzip", "lrzip",
                             "lrzip", "ffmpeg", "ffmpeg", "ffmpeg", "ffmpeg", "lrzip", "ffmpeg", "ffmpeg", "ffmpeg",
                             "ffmpeg", "ffmpeg", "ffmpeg", "ffmpeg", "lrzip", "ffmpeg", "lrzip", "ffmpeg"]
            _request_subject_map = dict(zip(_request_element_list, _subject_type))
        else:
            _subject_type = ["Workload", "lrzip", "ffmpeg", "lrzip", "ffmpeg", "ffmpeg", "lrzip", "lrzip", "lrzip", "ffmpeg",
                             "ffmpeg"]
            _request_subject_map = dict(zip(_request_element_list, _subject_type))

    if mix_status:
        _change_list = generate_change_element_set_from_model(_model_dict, _config_space, _request_element_list,
                                                              _simplize_request_element_list,
                                                              subject_type_list=_request_subject_map)
    else:
        _change_list = generate_change_element_set_from_model(_model_dict, _config_space, _request_element_list,
                                                              _simplize_request_element_list)

    _rank_list, _mean_st_list = qpme_running(_change_list, _qpme_file_location=_qpme_file)
    return _change_list, _rank_list, _mean_st_list


if __name__ == '__main__':
    local = True
    if local:
        random.seed(42)
        work_model_address = "./muBench_changed_files/teastore/WorkModel.json"
        tc = TestCases(work_model_address)
        tc.load_workload_model()
        tc.get_config_space()
        save_data = []
        for index in range(0, 50):
            change_elements, ranked_list, mean_st_ranked_list = qpme_process("ori/final_model.pkl", tc.config_space, index, type="train")
            data_package = {
                "change_elements": change_elements,
                "ranked_list": ranked_list,
                "mean_st_ranked_list": mean_st_ranked_list
            }
            # change_list
            # "element": element,
            # "mean": f"{pred_mean:.2f}",
            # "std": str(pred_std),
            # "config": _filted_config,
            print(data_package)
            save_data.append(data_package)
        print(len(save_data))
        with open(f"./qpme/predict_results/queue_predicted_results.pkl", "wb") as f:
            pickle.dump(save_data, f)
    else:
        generate_model()
