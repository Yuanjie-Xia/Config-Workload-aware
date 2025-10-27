"""
Runs service intensity experiment and outputs results.

It tests the gateway_agg aggregator deployment with a heterogeneous load to
identify the behaviours in CPU utilization and response delay.
The system deploys four services, three "worker" services (s1, s2, s3) and one
gateway_agg aggregator service which handles aggregate requests. In this context,
a heterogeneous load refers to the weight that is put on the worker services.
As such, requests can be S1-intensive and S3-intensive, respectively describing 
the amount of computation required by either service (i.e., s1 intensive requests 
require a lot of computation by s1 and little by s3). Here, S2 remains unaffected.

This script itself does nothing regarding the deployment topology (meaning that it
does not deploy the three services itself). The muBench deployment / setup is
defined in the various json files contained in this file's containing folder.

This script can be ran with various parameters. For these refer to `./many-experiment.sh`
and the `args_helper`.

The main lifecycle of this script is as follows:
1. Generate template files based on the passed parameters.
2. Iterate through the experiment `steps` number of times (i.e., the number of s1/s3-intensity
   configurations that is considerd; in some sense the granularity of the study.)
3. Create temporary runner parameters (workload etc.) and redeploy the various services.
4. Run the experiment (contained in `experiment_helper` as this is re-used in other experiments.)
5. Calculate some basic statistics.
6. Clean up temporary files.
"""

import os
import random
import dotenv

import gssi_experiment.util.doc_helper as doc_helper
import gssi_experiment.util.experiment_helper as exp_helper
import gssi_experiment.util.args_helper as args_helper
import gssi_experiment.util.util as util
import gssi_experiment.util.node_selector_helper as ns_helper
import gssi_experiment.util.tmp_exp_doc_helper as tmp_doc_helper
import gssi_experiment.util.k8s_helper as k8s_helper

dotenv.load_dotenv()

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
print(f"{BASE_FOLDER=}")

parser = args_helper.init_args(BASE_FOLDER)
parser.add_argument(
    "--aggregator-service-path",
    action="store",
    dest="aggregator_service_path",
    default=f"{BASE_FOLDER}/gateway_aggregator_service/service.yaml",
)
args = parser.parse_args()
args.simulation_steps = max(args.simulation_steps, 1)

random.seed(args.seed)
print(f"{args.seed=}")


def write_tmp_ga_service_for_node_selector_and_replicas_one_target(
    target_node: str,
) -> str:
    """Overwrites the gateway_agg aggregator service to use a node selector."""
    tmp_aggregator_service_path = f"{args.aggregator_service_path}.tmp"

    doc_helper.write_concrete_data_document(
        args.aggregator_service_path,
        tmp_aggregator_service_path,
        overwritten_fields=[
            (
                [3, "spec", "template", "spec"],
                {"nodeSelector": {"kubernetes.io/hostname": target_node}},
            ),
            ([3, "spec", "replicas"], args.replicas),
        ],
        editor_type=doc_helper.YamlEditor,
    )

    return tmp_aggregator_service_path


def write_tmp_ga_service_for_node_selector_and_replicas_multiple_targets(
    target_nodes: list,
) -> str:
    """Overwrites the gateway_agg aggregator service to use node affinity and spread constraints."""
    tmp_aggregator_service_path = f"{args.aggregator_service_path}.tmp"
    doc_helper.write_concrete_data_document(
        args.aggregator_service_path,
        tmp_aggregator_service_path,
        overwritten_fields=[
            (
                [3, "spec", "template", "spec", "affinity"],
                {
                    "nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [
                                {
                                    "matchExpressions": [
                                        {
                                            "key": "kubernetes.io/hostname",
                                            "operator":"In",
                                            "values": target_nodes,
                                        }
                                    ]
                                }
                            ]
                        }
                    },
                },
            ),
            (
                [3, "spec", "template", "spec", "affinity", "podAntiAffinity"],
                {
                    "requiredDuringSchedulingIgnoredDuringExecution": [
                        {
                            "labelSelector": {
                                "matchExpressions": [
                                    {
                                        "key": "app",
                                        "operator": "In",
                                        "values": ["gateway_agg-aggregator"],  # Replace with your app label
                                    }
                                ]
                            },
                            "topologyKey": "kubernetes.io/hostname",
                        }
                    ]
                },
            ),
            (
                [3, "spec", "template", "spec", "topologySpreadConstraints"],
                [
                    {
                        "maxSkew": 1,
                        "topologyKey": "kubernetes.io/hostname",
                        "whenUnsatisfiable": "ScheduleAnyway",
                        "labelSelector": {
                            "matchLabels": {"type": "gateway_agg-aggregator"}
                        },
                    }
                ],
            ),
            ([3, "spec", "replicas"], args.replicas),
        ],
        editor_type=doc_helper.YamlEditor,
    )
    return tmp_aggregator_service_path


def write_tmp_ga_service_for_node_selector_and_replicas() -> str:
    """Sets the target node field inside the deployment yaml and outputs it to a temp file."""

    target_nodes = ns_helper.get_target_nodes(args)
    if len(target_nodes) == 0:
        return write_tmp_ga_service_for_node_selector_and_replicas_one_target(
            target_nodes[0]
        )
    else:
        return write_tmp_ga_service_for_node_selector_and_replicas_multiple_targets(
            target_nodes
        )


def run_the_experiment():
    """Does what it says."""

    mubench_k8s_template_folder = os.path.dirname(BASE_FOLDER)

    # Overwrites affinity in GA service and muBench service yamls.
    tmp_ga_service_yaml_path = write_tmp_ga_service_for_node_selector_and_replicas()
    ns_helper.load_and_write_node_affinity_template(args, mubench_k8s_template_folder)
    k8s_param_path = os.path.dirname(__file__) + "/K8sParameters.json"
    k8s_params_file_path = f"{k8s_param_path}.tmp"
    tmp_doc_helper.write_tmp_k8s_params(
        k8s_param_path, k8s_params_file_path, args.cpu_limit, args.replicas
    )

    # Executes experiments for every considered s1 intensity value.
    for step_idx in util.shuffled_range(0, args.simulation_steps + 1, 1):
        tmp_runner_param_file_path = (
            tmp_doc_helper.write_tmp_runner_params_for_simulation_step(
                step_idx, args.simulation_steps, args.base_runner_param_file_name
            )
        )
        k8s_helper.safe_apply_k8s_yaml_file(tmp_ga_service_yaml_path)
        exp_params = exp_helper.ExperimentParameters(
            k8s_params_file_path,
            tmp_runner_param_file_path,
            mubench_k8s_template_folder,
            exp_helper.get_output_folder(BASE_FOLDER, args.name, step_idx),
            args.wait_for_pods_delay,
        )
        exp_helper.run_experiment(args, exp_params)


if __name__ == "__main__":
    run_the_experiment()
