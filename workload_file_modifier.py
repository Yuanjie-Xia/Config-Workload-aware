import json


def search_json_keys(obj, search_term, path="", found_keys=None):
    if found_keys is None:
        found_keys = []  # Initialize the list if it's not passed in

    # If it's a dictionary, recurse through each key-value pair
    if isinstance(obj, dict):
        for key, value in obj.items():
            # Check if the key contains the search term
            if search_term.lower() in key.lower():
                found_keys.append(f"{path}.{key}" if path else key)
            # Recurse through the value, passing along the path
            search_json_keys(value, search_term, f"{path}.{key}" if path else key, found_keys)

    # If it's a list, recurse through each item by index
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            search_json_keys(item, search_term, f"{path}[{index}]", found_keys)

    return found_keys


class TestCases:
    def __init__(self, work_model_address):
        self.work_model_address = work_model_address# "Original_Configs/WorkModel.json"
        self.config_values = {
            'algorithm': ['-z', '-b', '-g', '-n', '-l'],
            'level': [8, 9],
            'window': [20, 40, 60, 80],
            'nice': [-14, -7, 0, 7, 14],
            'processor': [1, 2, 3, 4],
            'codec': ['libx264', 'libx265'],
            'preset': ['ultrafast', 'superfast', 'faster', 'fast', 'medium', 'slow', 'slower'],
            # 'preset': ['ultrafast', 'faster', 'medium', 'slower'],
            'crf': [11, 16, 21, 26, 31, 36, 41, 46, 51],
            # 'crf': [11, 21, 31, 41, 51],
            'cache_size': ['2000', '3000', '4000'],
            'page_size': ['512', '1024', '2048'],
            'autovacuum': [True, False],
            'exclusive': [True, False],
            'journal_mode': ['MEMORY', 'WAL', 'OFF'],
            'nosync': [True, False],
            'stats': [True, False]
        }
        # 'workers': [4, 5, 6, 7, 8, 9, 10, 11, 12],
        # 'threads': [64, 128, 256]
        self.configurations = list(self.config_values.keys())
        self.config_space = None
        self.workload_json = None

    def load_workload_model(self):
        with open(self.work_model_address, 'r') as _file:
            _json_data = json.load(_file)
        self.workload_json = _json_data

    def get_config_space(self, target_pod=None, internal_service=None):
        found_keys = []
        for item in self.configurations:
            sub_found_keys = search_json_keys(self.workload_json, item)
            if len(sub_found_keys) > 0:
                found_keys.append(sub_found_keys)
        available_values = {}
        for group in found_keys:
            for path in group:
                if target_pod is not None and not path.startswith(f"{target_pod}."):
                    continue  # Skip keys not in the target pod
                if internal_service is not None and not path.__contains__(f"{internal_service}"):
                    continue
                key_suffix = path.split('.')[-1]
                if key_suffix in self.config_values:
                    available_values[path] = self.config_values[key_suffix]
        self.config_space = available_values

    def modify_workload_model(self, config_need_modify, config_value4modify):
        keys = config_need_modify.split(".")
        # Start from the root json object
        obj = self.workload_json
        # Traverse down to the final nested object
        for key in keys[:-1]:  # Go until the second-to-last key
            obj = obj.setdefault(key, {})
        # Set the final key to the new value
        obj[keys[-1]] = config_value4modify

    def save_workload_model(self, save_file_path):
        try:
            with open(save_file_path, 'w') as f:
                json.dump(self.workload_json, f, indent=4)
            print(f"Workload model saved successfully to {save_file_path}")
        except Exception as e:
            print(f"Failed to save workload model: {e}")


def main():
    tc = TestCases("muBench_changed_files/WorkModel_test.json")
    tc.load_workload_model()
    tc.get_config_space()
    print(tc.config_space)


if __name__ == '__main__':
    main()




