"""
Implements some node selector utility methods.
They load affinity settings using commandline arguments and write that to yaml files.
"""

from os import getenv
from argparse import Namespace

from typing import List


def add_extra_tabs(text: str, extra_tabs: int) -> str:
    new_tab = "  " * extra_tabs
    return text.replace("\n", f"\n{new_tab}")


def get_target_nodes(args: Namespace) -> List[str]:
    """Gets target nodes."""
    target_nodes = [node for node in args.node_selector.split(",") if len(node) > 0]
    # HACK: This shouldn't be handled with an env variable; it should check for "existing" nodes in the cluster and remove those that aren't present.
    if (
        getenv("USE_MINIKUBE", "false").lower() == "false"
        and "minikube" in target_nodes
    ):
        print("Not running in minikube mode, removing it from the possible targets.")
        target_nodes.remove("minikube")
    print(f"{target_nodes=}")
    return target_nodes


def get_node_affinity_template(args: Namespace) -> str:
    """Generates node affinity template based on command line arguments."""
    target_nodes = get_target_nodes(args)
    if len(target_nodes) == 0:
        return ""

    # If it's only one, it will force it to be scheduled on that node.
    # Using spread constraints incidentally doesn't work here.
    if len(target_nodes) == 1:
        node_selector_template = f"""
      nodeSelector:
        kubernetes.io/hostname: {target_nodes[0]}
        """
        return node_selector_template

    equal_distribution_template = """
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
              nodeSelectorTerms:
              - matchExpressions:
                - key: kubernetes.io/hostname
                  operator: In
                  values:{target_nodes}
    """
    target_node_template = """
                    - {name}"""

    target_nodes = [target_node_template.format(name=node) for node in target_nodes]

    target_nodes = "".join(target_nodes)
    equal_distribution_template = equal_distribution_template.format(
        target_nodes=target_nodes
    )
    equal_distribution_template = ""
    # Adding this 'forces' pods to be spread across different nodes
    # (this is a bugged k8s feature though as it works during the initial deployment
    # but not during re-deployments; i.e., if you want to ensure this, you have to delete
    # a deployment, wait for it to be gone, and only then re-apply it; this is a known bug,
    # but k8s isn't prioritizing fixing it).
    topology_spread_template = """
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: ScheduleAnyway
          labelSelector:
            matchLabels:
              type: {{SERVICE_NAME}}
    """
    equal_distribution_template += topology_spread_template

    node_selector_template = equal_distribution_template

    return node_selector_template


def write_tmp_deployment_template(
    base_folder: str, deployment_template_file: str, selector_template: str
):
    """Writes a temporary k8s deployment configuration that specifies the node affinity settings."""
    # Reads template base.
    deployment_path = f"{base_folder}/Templates/{deployment_template_file}"
    with open(deployment_path, "r", encoding="utf-8") as base_file:
        data = base_file.read()

    setting = selector_template if selector_template else ""
    data = data.replace("{{NODE_AFFINITY}}", setting)
    # writes data
    deployment_path = f"{base_folder}/Templates/DeploymentTemplate.yaml"
    with open(deployment_path, "w+", encoding="utf-8") as output_file:
        output_file.write(data)


def load_and_write_node_affinity_template(
    args: Namespace,
    base_folder: str,
    deployment_template_file: str = "muBench-DeploymentTemplate-Base.yaml",
):
    """
    Loads the used affinity template applies it to the specified deployment
    file and outputs it to the corresonding muBench file.
    """
    node_affinity_setting = get_node_affinity_template(args)
    add_extra_tabs(node_affinity_setting, 0)
    write_tmp_deployment_template(
        base_folder, deployment_template_file, node_affinity_setting
    )
