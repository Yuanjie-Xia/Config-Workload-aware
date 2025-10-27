import itertools
import pickle
import random
import statistics

import numpy as np

from collections import defaultdict
from single_pod_model import encode_config
from workload_file_modifier import TestCases
from analysis_spike import change_qpme, qpme_running

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
_subject_type = ["Workload", "ffmpeg", "lrzip", "ffmpeg", "ffmpeg", "ffmpeg", "lrzip", "lrzip", "lrzip",
                 "lrzip", "ffmpeg", "ffmpeg", "ffmpeg", "ffmpeg", "lrzip", "ffmpeg", "ffmpeg", "ffmpeg", "ffmpeg",
                 "ffmpeg", "ffmpeg", "ffmpeg", "lrzip", "ffmpeg", "lrzip", "ffmpeg"]
_request_subject_map = dict(zip(_simplize_request_element_list, _subject_type))

_servers = ["s0", "s1", "s2", "s3", "s4"]
_url_type = ["url_1", "url_2", "url_3", "url_4", "url_5", "url_6"]


def generate_config_lazy(config_options):
    keys = list(config_options.keys())
    for values in itertools.product(*(config_options[key] for key in keys)):
        yield dict(zip(keys, values))


def get_box_values(model_file_address="./ori/final_model.pkl", subject="ffmpeg"):
    work_model_address = "./muBench_changed_files/teastore_mix/WorkModel.json"
    tc = TestCases(work_model_address)
    tc.load_workload_model()
    # tc.get_config_space()
    tc.get_config_space("s4", f"{subject}_test_only")
    with open(model_file_address, "rb") as f:
        _final_model = pickle.load(f)

    perf_list = []
    for config in generate_config_lazy(tc.config_space):
        # _filted_config, _filter_config_space = reshape_config_space(config, tc.config_space, i, simplized_elements)
        X_encoded = np.array([encode_config(config, tc.config_space)])
        pred_mean = float(_final_model.predict(X_encoded)[0])
        perf_list.append(pred_mean)

    perf_list = np.array(perf_list)

    # Step 1: Compute histogram
    counts, bin_edges = np.histogram(perf_list, bins=8)

    # Step 2: Compute bin centers (mean of bin edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers


_bin_centers = get_box_values()
_lrzip_bin_centers = get_box_values(model_file_address="./ori/lrzip_final_model.pkl", subject="lrzip")
bin_center_dict = {
    "lrzip": _lrzip_bin_centers,
    "ffmpeg": _bin_centers
}
frequency_list = list(range(10, 76, 10))
_prob_list_0 = [1] * len(_url_type)   # Workload elements
_prob_list_3 = [1] * len(frequency_list)


def normalize(prob_list):
    total = sum(prob_list)
    return [p / total for p in prob_list]


def weighted_sample(items, probs):
    return random.choices(items, weights=probs, k=1)[0]


def update_probs(element, feedback_score, alpha=0.05):
    element *= (1 + alpha * (feedback_score - 0.5))  # feedback ∈ [0,1]
    return element


def generate_weighted_mean_list(elements, bin_centers, prob_3d, init_status=False):
    mean_list = []
    for elem in elements:
        if elem == "Workload":
            continue  # skip workload

        try:
            server, url_type = elem.split("/")
        except ValueError:
            raise ValueError(f"Invalid element format: {elem}")

        subject_name = _request_subject_map[elem]
        selected_bin_centers = bin_centers[subject_name]
        server_idx = _servers.index(server)
        url_idx = _url_type.index(url_type)
        sub_bin_prob_list = prob_3d[server_idx, url_idx, :]
        if init_status:
            sample_value = random.choice(selected_bin_centers)
        else:
            sample_value = weighted_sample(selected_bin_centers, sub_bin_prob_list)
        mean_list.append(sample_value)

    # Need to use prob list 0, 1, 2, 3
    return mean_list


def build_masked_3d_prob(_servers, _url_type, _simplize_request_element_list, num_values):
    S = len(_servers)
    U = len(_url_type)
    V = num_values

    # Start with all ones
    prob_3d = np.ones((S, U, V), dtype=float)

    # Build set of valid "sX/url_Y" combinations
    valid_combinations = set(elem for elem in _simplize_request_element_list if elem != "Workload")

    # Set invalid server-url pairs to 0 across all values
    for i, server in enumerate(_servers):
        for j, url in enumerate(_url_type):
            key = f"{server}/{url}"
            if key not in valid_combinations:
                prob_3d[i, j, :] = 0.0
            else:
                if _request_subject_map[key] == "ffmpeg":
                    prob_3d[i, j, 8:16] = 0.0
                if _request_subject_map[key] == "lrzip":
                    prob_3d[i, j, 0:8] = 0.0
    return prob_3d


_prob_3d = build_masked_3d_prob(_servers, _url_type, _simplize_request_element_list, len(_bin_centers) + len(_lrzip_bin_centers))


def simulate_feedback(_mean_st_list, elements):
    # Skip "Workload", so feedback_list has same length as _mean_st_list
    feedback_list = [0] * (len(elements) - 1)

    name_to_index = {name: i - 1 for i, name in enumerate(elements) if name != "Workload"}

    for elem_name, mean_st in _mean_st_list:
        if mean_st <= 300:
            if elem_name in name_to_index:
                idx = name_to_index[elem_name]
                feedback_list[idx] = 1
    min_element_list = _mean_st_list[-4:-1]
    min_index_list = []
    for min_element in min_element_list:
        min_index = elements.index(min_element[0])
        min_index_list.append(min_index)
    return feedback_list, min_index_list


def generate_weighted_frequency(freq_list, prob_weights):
    normalized_probs = normalize(prob_weights)
    return weighted_sample(freq_list, normalized_probs)


def mutate_child(mean_list, element_list, min_id_list, prob_3d, bin_center_dict):
    for min_id in min_id_list:
        element_name = element_list[min_id]
        server, url_type = element_name.split("/")
        mutate_number = 0
        for element in element_list:
            if element == "Workload":
                continue
            subject_name = _request_subject_map[element]
            if subject_name == "ffmpeg":
                active_range = list(range(0, 8))
            if subject_name == "lrzip":
                active_range = list(range(8, 16))
            selected_bin_center = bin_center_dict[subject_name]
            element_server, element_url_type = element.split("/")
            server_idx = _servers.index(element_server)
            url_idx = _url_type.index(element_url_type)
            random_mutate = random.randint(0, 10)
            if server in element and random_mutate > 7:
                element_idx = element_list.index(element)
                mean_list[element_idx-1] = weighted_sample(selected_bin_center, prob_3d[server_idx, url_idx, active_range])
                mutate_number += 1
            if url_type in element and random_mutate > 7:
                element_idx = element_list.index(element)
                mean_list[element_idx-1] = weighted_sample(selected_bin_center, prob_3d[server_idx, url_idx, active_range])
                mutate_number += 1
            if mutate_number >= 1:
                break
        if mutate_number < 1:
            random_element = random.randrange(len(mean_list))
            random_name = element_list[random_element + 1]
            subject_name = _request_subject_map[random_name]
            if subject_name == "ffmpeg":
                active_range = list(range(0, 8))
            if subject_name == "lrzip":
                active_range = list(range(8, 16))
            selected_bin_center = bin_center_dict[subject_name]
            random_value = random.choice(selected_bin_center)
            mean_list[random_element] = random_value
    return mean_list


def check_seed_list(min_idx_list, min_feedback, seed_list):
    # Build a set of all seen (idx_list, feedback) pairs
    seen = {
        (tuple(seed['min_idx_list']), seed['min_feedback'])
        for seed in seed_list
        if 'min_idx_list' in seed and 'min_feedback' in seed
    }

    current = (tuple(min_idx_list), min_feedback)
    is_repeat = current in seen

    print(f"seen={seen}, current={current}, is_repeat={is_repeat}")
    return is_repeat


def std_relative_change(feedback_std, init_feedback_std):
    judge = False
    if init_feedback_std > 1e-8:
        relative_diff = (feedback_std-init_feedback_std)/init_feedback_std
        if relative_diff > 0.2:
            judge = True
    else:
        diff = feedback_std - init_feedback_std
        if diff > 0.5:
            judge = True
    return judge


def generate_and_update(n_cases=20, children_limit=20):
    global _prob_3d, _prob_list_3
    # frequency = random.choice(frequency_list)
    save_data = []
    seed_list = []
    seen_seed_set = set()
    for i in range(n_cases):
        frequency = frequency_list[i % len(frequency_list)]
        elements = _simplize_request_element_list
        mean_list = [0]
        element_str = ""
        times = 0
        while True:
            if times > 100:
                print("Exceed looping time.")
                break
            random.seed(times)
            times += 1
            # frequency = weighted_sample(frequency_list, _prob_list_3)
            mean_list = generate_weighted_mean_list(elements, bin_center_dict, _prob_3d, init_status=True)
            # must be larger than first bin
            if max(mean_list) <= min(_bin_centers[0], _lrzip_bin_centers[0]):
                continue
            combined = [*mean_list, frequency]
            combined_rounded = [round(float(x), 2) for x in combined]
            element_str = "[" + ",".join(f"{v:.2f}" for v in combined_rounded) + "]"
            # skip repeats
            if element_str in seen_seed_set:
                continue
            # valid & new → accept
            break

        seen_seed_set.add(element_str)
        change_list = change_qpme(mean_list, _request_element_list, frequency)

        _rank_list, _mean_st_list = qpme_running(change_list)
        data_package = {
            "case_id": i,
            "seed_id": 0,
            "change_elements": change_list,
            "ranked_list": _rank_list,
            "mean_st_ranked_list": _mean_st_list,
            "prob_3d": _prob_3d.copy(),
            "prob_workload": _prob_list_3.copy()
        }
        save_data.append(data_package)
        feedback_score_dict, min_idx_list = simulate_feedback(_mean_st_list, _request_element_list)
        seed_mean_list = mean_list
        init_feedback_std = statistics.stdev(feedback_score_dict)
        seed = {
            "seed_mean_list": seed_mean_list,
            "min_idx_list": min_idx_list,
            "min_feedback": min(feedback_score_dict)
        }
        seed_list.append(seed)
        branch_seed_list = [seed]
        loop_time = 0
        while loop_time < len(branch_seed_list):
            seed_mean_list = branch_seed_list[loop_time]["seed_mean_list"]
            min_idx_list = branch_seed_list[loop_time]["min_idx_list"]
            init_min_idx = branch_seed_list[loop_time]["min_idx_list"]
            print(f"Running seed: {seed_mean_list}, case_id is {i}")

            exceed_time = 0
            if exceed_time > 100:
                print("Time exceed 100.")
                break
            for child_id in range(children_limit):
                element_str = ""
                seed_times = 0
                exceed_status = False
                while True:
                    if seed_times > 100:
                        print(_bin_centers[0])
                        print(seed_mean_list)
                        print("Exceed looping time.")
                        exceed_status = True
                        exceed_time += 1
                        break
                    random.seed(seed_times)
                    seed_times += 1
                    # frequency = weighted_sample(frequency_list, _prob_list_3)
                    seed_mean_list = mutate_child(seed_mean_list, elements, min_idx_list, _prob_3d, bin_center_dict)
                    # must be larger than first bin
                    if max(seed_mean_list) <= _bin_centers[0]:
                        print("break from the basic filter")
                        continue
                    combined = [*seed_mean_list, frequency]
                    combined_rounded = [round(float(x), 2) for x in combined]
                    element_str = "[" + ",".join(f"{v:.2f}" for v in combined_rounded) + "]"
                    # skip repeats
                    if element_str in seen_seed_set:
                        print(f"Find {element_str} in set.")
                        continue
                    # valid & new → accept
                    break

                if exceed_status:
                    continue

                seen_seed_set.add(element_str)

                change_list = change_qpme(seed_mean_list, _request_element_list, frequency)
                _rank_list, _mean_st_list = qpme_running(change_list)

                feedback_score_dict, min_idx_list = simulate_feedback(_mean_st_list, _request_element_list) # feedback=1 when time<300
                feedback_std = statistics.stdev(feedback_score_dict)

                if ~check_seed_list(min_idx_list, min(feedback_score_dict), seed_list) and min(
                        feedback_score_dict) > 0 and len(branch_seed_list) < 20:
                    print(
                        f"ADDED NEW SEED. STD: {init_feedback_std}, {feedback_std}; IDX: {init_min_idx}, {min_idx_list}, min feedback: {min(feedback_score_dict)}, length is {len(branch_seed_list)}")
                    new_seed = {
                        "seed_mean_list": seed_mean_list,
                        "min_idx_list": min_idx_list,
                        "min_feedback": min(feedback_score_dict)
                    }
                    branch_seed_list.append(new_seed)
                    seed_list.append(new_seed)
                    init_feedback_std = max(feedback_std, init_feedback_std)
                for idx, e in enumerate(elements):
                    workload_feedback = 0
                    if e == "Workload":
                        idx = frequency_list.index(frequency)
                        max_feedback = max(feedback_score_dict)
                        if max_feedback < 1:
                            workload_feedback = 0
                        else:
                            workload_feedback = 1
                        _prob_list_3[idx] = update_probs(_prob_list_3[idx], workload_feedback, alpha=0.05)
                    # Need an update probs here
                    else:
                        e_id = elements.index(e)
                        feedback_score = feedback_score_dict[e_id-1]
                        feedback_score = (feedback_score + workload_feedback)/2
                        url = e.split("/")[1]
                        server = e.split("/")[0]
                        subject = _request_subject_map[e]
                        selected_bin_center = bin_center_dict[subject]
                        if subject == "ffmpeg":
                            value_idx = list(selected_bin_center).index(seed_mean_list[idx - 1])
                        else:
                            value_idx = list(selected_bin_center).index(seed_mean_list[idx - 1]) + 8
                        server_idx = _servers.index(server)
                        url_idx = _url_type.index(url)
                        _prob_3d[server_idx, url_idx, value_idx] = update_probs(
                            _prob_3d[server_idx, url_idx, value_idx], feedback_score, alpha=0.1)
                        '''for u_id, _ in enumerate(_prob_3d[server_idx, :, :]):
                            for v_id, val in enumerate(_prob_3d[server_idx, u_id, :]):
                                if val > 0:
                                    _prob_3d[server_idx, u_id, v_id] = update_probs(_prob_3d[server_idx, u_id, v_id], feedback_score, alpha=0.05)
                        for s_id, _ in enumerate(_prob_3d[:, url_idx, :]):
                            for v_id, val in enumerate(_prob_3d[s_id, url_idx, :]):
                                if val > 0:
                                    _prob_3d[s_id, url_idx, v_id] = update_probs(_prob_3d[s_id, url_idx, v_id],
                                                                               feedback_score, alpha=0.05)'''

                data_package = {
                    "case_id": i,
                    "seed_id": loop_time,
                    "change_elements": change_list,
                    "ranked_list": _rank_list,
                    "mean_st_ranked_list": _mean_st_list,
                    "prob_3d": _prob_3d.copy(),
                    "prob_workload": _prob_list_3.copy()
                }
                save_data.append(data_package)
                print("--------------------------------------------------------")
                with open(f"./qpme/simulation_results.pkl", "wb") as f:
                    pickle.dump(save_data, f)
            loop_time += 1


def main():
    generate_and_update(n_cases=len(frequency_list), children_limit=100)


if __name__ == '__main__':
    main()
