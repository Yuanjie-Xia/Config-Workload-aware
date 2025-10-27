"""
Runs service intensity experiment and outputs results.
It accounts for heterogeneous requests and different amounts of tasks offloaded to the gateway.
"""

import os
import random
import itertools
from typing import Iterator, Tuple

import gssi_experiment.util.doc_helper as doc_helper
import gssi_experiment.util.experiment_helper as exp_helper
import gssi_experiment.util.args_helper as args_helper
import gssi_experiment.util.util as util
import gssi_experiment.util.node_selector_helper as ns_helper
import gssi_experiment.util.tmp_exp_doc_helper as tmp_doc_helper


BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
print(f"{BASE_FOLDER=}")


parser = args_helper.init_args(BASE_FOLDER)
parser.add_argument(
    "-gw",
    "--gateway-load",
    action="store",
    dest="gateway_load_range",
    default="[0,10]",
    help="The number of simulations that are performed w.r.t. gateway offloading.",
)
args = parser.parse_args()
args.simulation_steps = max(args.simulation_steps, 1)

random.seed(args.seed)
print(f"{args.seed=}")


def write_tmp_work_model_for_offload(gw_offload: int) -> str:
    """Overwrites the "trials" field in the workmodel."""
    # Defines workloads per service.
    service_workloads = [("s1", 20), ("s2", 12), ("s3", 15)]
    base_nested_key = [
        "__service",
        "internal_service",
        "loader",
        "cpu_stress",
        "trials",
    ]

    def service_workload_nested_key_generator():
        """Generates the nested key/value tuples necessary to overwrite the WorkModel."""
        for service, workload in service_workloads:
            base_nested_key[0] = service
            workload = workload - gw_offload
            yield base_nested_key.copy(), workload

    tmp_base_worker_model_path = f"{args.base_worker_model_file_name}.tmp"
    doc_helper.write_concrete_data_document(
        source_path=args.base_worker_model_file_name,
        target_path=tmp_base_worker_model_path,
        overwritten_fields=[
            (
                ["gw", "internal_service", "loader", "cpu_stress", "trials"],
                gw_offload,
            ),
            *service_workload_nested_key_generator(),
        ],
        editor_type=doc_helper.JsonEditor,
    )
    return tmp_base_worker_model_path


def get_experimental_config_iterator() -> Iterator[Tuple[int, int]]:
    """Returns all possible tested configurations."""
    sim_steps = util.shuffled_range(0, args.simulation_steps + 1, 1)
    (gw_min, gw_max, gw_step) = (
        int(ele) for ele in args.gateway_load_range[1:-1].split(",")
    )
    gateway_steps = util.shuffled_range(gw_min, gw_max + 1, gw_step)
    print(f'{gateway_steps=}')
    return itertools.product(gateway_steps, sim_steps)


def build_output_folder_path(step_idx, gateway_offload) -> str:
    """factory for the output path."""
    output_folder = (
        exp_helper.get_output_folder(BASE_FOLDER, args.name, step_idx)
        + f"/{gateway_offload}_offload/"
    )
    return output_folder


def run_the_experiment():
    """Does what it says."""

    # Overwrites work model and k8s params file.
    k8s_param_path = os.path.dirname(__file__) + "/K8sParameters.json"
    k8s_params_file_path = f"{k8s_param_path}.tmp"
    tmp_doc_helper.write_tmp_k8s_params(
        k8s_param_path, k8s_params_file_path, args.cpu_limit, args.replicas
    )

    mubench_k8s_template_folder = os.path.dirname(BASE_FOLDER)
    ns_helper.load_and_write_node_affinity_template(args, mubench_k8s_template_folder)

    # Iterates through all possible experimental configurations.
    for gateway_offload, step_idx in get_experimental_config_iterator():
        print(f"{gateway_offload=}, {step_idx=}")

        # Updates configuration files according to the experimental settings.
        tmp_runner_param_file_path = (
            tmp_doc_helper.write_tmp_runner_params_for_simulation_step(
                step_idx, args.simulation_steps, args.base_runner_param_file_name
            )
        )
        write_tmp_work_model_for_offload(gateway_offload)

        # Runs the experiment with the given parameters.
        exp_params = exp_helper.ExperimentParameters(
            k8s_params_file_path,
            tmp_runner_param_file_path,
            mubench_k8s_template_folder,
            build_output_folder_path(step_idx, gateway_offload),
            args.wait_for_pods_delay,
        )
        exp_helper.run_experiment(args, exp_params)


if __name__ == "__main__":
    run_the_experiment()
