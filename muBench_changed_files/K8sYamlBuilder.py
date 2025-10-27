import os
import yaml


DEFAULT_K8s_YAML_BUILDER_PATH = os.path.dirname(os.path.abspath(__file__))

SIDECAR_TEMPLATE = "- name: %s-sidecar\n          image: %s"
NODE_AFFINITY_TEMPLATE = {
    "affinity": {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {
                        "matchExpressions": [
                            {
                                "key": "kubernetes.io/hostname",
                                "operator": "In",
                                "values": [""],
                            }
                        ]
                    }
                ]
            }
        }
    }
}
POD_ANTIAFFINITI_TEMPLATE = {
    "affinity": {
        "podAntiAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": [
                {
                    "labelSelector": {
                        "matchExpressions": [
                            {"key": "app", "operator": "In", "values": [""]}
                        ]
                    },
                    "topologyKey": "kubernetes.io/hostname",
                }
            ]
        }
    }
}


# Add params to work_model json from kubernetes paramenters
def customization_work_model(workmodel, k8s_parameters):
    for service in workmodel:
        # Skips entries that are not generated services.
        is_generated_key = "is_generated"
        if (
            is_generated_key in workmodel[service]
            and not workmodel[service][is_generated_key]
        ):
            print(f"###############\nSkipping service {service}.\n###############")
            continue

        workmodel[service].update(
            {
                "url": f"{service}.{k8s_parameters['namespace']}.svc.{k8s_parameters['cluster_domain']}.local"
            }
        )
        if service not in k8s_parameters["image"]:
            service_name = "Other"
        else:
            service_name = service
        workmodel[service].update({"path": k8s_parameters["path"]})
        workmodel[service].update({"image": k8s_parameters["image"][service_name]})
        workmodel[service].update({"namespace": k8s_parameters["namespace"]})
        # TODO: These should be defined per service, not in general.
        if "replicas" in k8s_parameters.keys():
            # override replica value of workmodel.json
            workmodel[service].update({"replicas": k8s_parameters["replicas"]})
        if "cpu-requests" in k8s_parameters.keys():
            # override cpu-requests value of workmodel.json
            workmodel[service].update({"cpu-requests": k8s_parameters["cpu-requests"]})
        if "cpu-limits" in k8s_parameters.keys():
            # override cpu-limits value of workmodel.json
            workmodel[service].update({"cpu-limits": k8s_parameters["cpu-limits"]})
        if "memory-requests" in k8s_parameters.keys():
            # override memory-requests value of workmodel.json
            workmodel[service].update(
                {"memory-requests": k8s_parameters["memory-requests"]}
            )
        if "memory-limits" in k8s_parameters.keys():
            # override memory-limits value of workmodel.json
            workmodel[service].update(
                {"memory-limits": k8s_parameters["memory-limits"]}
            )
    print("Work Model Updated!")


def create_deployment_yaml_files(
    workmodel, k8s_parameters, nfs, output_path, k8s_yaml_builder_path
):
    if k8s_yaml_builder_path is None:
        k8s_yaml_builder_path = DEFAULT_K8s_YAML_BUILDER_PATH

    print(f"Using YAML builder path '{k8s_yaml_builder_path}'.")

    namespace = k8s_parameters["namespace"]
    counter = 0
    for service in workmodel:
        # Skips entries that are not generated services.
        is_generated_key = "is_generated"
        if (
            is_generated_key in workmodel[service]
            and not workmodel[service][is_generated_key]
        ):
            print(f"###############\nSkipping service {service}.\n###############")
            continue

        counter = counter + 1
        with open(
            f"{k8s_yaml_builder_path}/Templates/DeploymentTemplate.yaml", "r"
        ) as file:
            f = file.read()
            f = f.replace("{{SERVICE_NAME}}", service)
            f = f.replace("{{IMAGE}}", workmodel[service]["image"])
            f = f.replace("{{NAMESPACE}}", namespace)
            if "sidecar" in workmodel[service].keys():
                f = f.replace(
                    "{{SIDECAR}}",
                    SIDECAR_TEMPLATE % (service, workmodel[service]["sidecar"]),
                )
            else:
                f = f.replace("{{SIDECAR}}", "".rstrip())
            if "replicas" in workmodel[service].keys():
                f = f.replace("{{REPLICAS}}", str(workmodel[service]["replicas"]))
            else:
                f = f.replace("{{REPLICAS}}", "1")
            if "node_affinity" in workmodel[service].keys():
                NODE_AFFINITY_TEMPLATE_TO_ADD = NODE_AFFINITY_TEMPLATE
                NODE_AFFINITY_TEMPLATE_TO_ADD["affinity"]["nodeAffinity"][
                    "requiredDuringSchedulingIgnoredDuringExecution"
                ]["nodeSelectorTerms"][0]["matchExpressions"][0].update(
                    {"values": workmodel[service]["node_affinity"]}
                )
                f = f.replace(
                    "{{NODE_AFFINITY}}",
                    str(yaml.dump(NODE_AFFINITY_TEMPLATE_TO_ADD))
                    .rstrip()
                    .replace("\n", "\n   "),
                )
            else:
                f = f.replace("{{NODE_AFFINITY}}", "")
            if (
                "pod_antiaffinity" in workmodel[service].keys()
                and workmodel[service]["pod_antiaffinity"] == True
            ):
                POD_ANTIAFFINITY_TO_ADD = POD_ANTIAFFINITI_TEMPLATE
                POD_ANTIAFFINITY_TO_ADD["affinity"]["podAntiAffinity"][
                    "requiredDuringSchedulingIgnoredDuringExecution"
                ][0]["labelSelector"]["matchExpressions"][0]["values"][0] = service
                POD_ANTIAFFINITY_TO_ADD = (
                    str(yaml.dump(POD_ANTIAFFINITY_TO_ADD))
                    .replace("\n", "\n        ")
                    .rstrip()
                )
                f = f.replace("{{POD_ANTIAFFINITY}}", POD_ANTIAFFINITY_TO_ADD)
            else:
                f = f.replace("{{POD_ANTIAFFINITY}}", "".rstrip())
            if "workers" in workmodel[service].keys():
                f = f.replace("{{PN}}", f'\'{workmodel[service]["workers"]}\'')
            else:
                f = f.replace("{{PN}}", "'1'")
            if "threads" in workmodel[service].keys():
                f = f.replace("{{TN}}", f'\'{workmodel[service]["threads"]}\'')
            else:
                f = f.replace("{{TN}}", "'4'")

            rank_string = ""  # ranck string is used to order the yaml file as a funciont of the cpu-requests
            if len(
                set(workmodel[service].keys()).intersection(
                    {"cpu-limits", "memory-limits", "cpu-requests", "memory-requests"}
                )
            ):
                s = ""
                if (
                    "cpu-requests" in workmodel[service].keys()
                    or "memory-requests" in workmodel[service].keys()
                ):
                    s = s + "\n            requests:"
                    if "cpu-requests" in workmodel[service].keys():
                        s = (
                            s
                            + "\n              cpu: "
                            + workmodel[service]["cpu-requests"]
                        )
                        if "m" in workmodel[service]["cpu-requests"]:
                            rank_string = str(
                                int(workmodel[service]["cpu-requests"].replace("m", ""))
                            ).zfill(5)
                        else:
                            rank_string = str(
                                int(float(workmodel[service]["cpu-requests"]) * 1000)
                            ).zfill(5)
                    if "memory-requests" in workmodel[service].keys():
                        s = (
                            s
                            + "\n              memory: "
                            + workmodel[service]["memory-requests"]
                        )
                if (
                    "cpu-limits" in workmodel[service].keys()
                    or "memory-limits" in workmodel[service].keys()
                ):
                    s = s + "\n            limits:"
                    if "cpu-limits" in workmodel[service].keys():
                        s = (
                            s
                            + "\n              cpu: "
                            + workmodel[service]["cpu-limits"]
                        )
                    if "memory-limits" in workmodel[service].keys():
                        s = (
                            s
                            + "\n              memory: "
                            + workmodel[service]["memory-limits"]
                        )
                f = f.replace("{{RESOURCES}}", s)
            else:
                f = f.replace("{{RESOURCES}}", "{}")
        if not os.path.exists(f"{output_path}/yamls"):
            os.makedirs(f"{output_path}/yamls")

        # rank used to sort the deployment so as more demanding PODs are deployed first
        with open(
            f"{output_path}/yamls/{str(rank_string).zfill(3)}-{k8s_parameters['prefix_yaml_file']}-{service}.yaml",
            "w",
        ) as file:
            file.write(f)

    with open(
        f"{k8s_yaml_builder_path}/Templates/ConfigMapNginxGwTemplate.yaml", "r"
    ) as file:
        f = file.read()
        f = f.replace("{{NAMESPACE}}", namespace)
        f = f.replace("{{PATH}}", k8s_parameters["path"])
        dns_resolver = os.getenv("K8S_DNS_RESOLVER", None)
        if dns_resolver is None:
            dns_resolver = k8s_parameters["dns-resolver"]
        print(f"{dns_resolver=}")
        f = f.replace("{{RESOLVER}}", dns_resolver)

    with open(f"{output_path}/yamls/ConfigMapNginxGw.yaml", "w") as file:
        file.write(f)

    with open(
        f"{k8s_yaml_builder_path}/Templates/DeploymentNginxGwTemplate.yaml", "r"
    ) as file:
        f = file.read()
        f = f.replace("{{NAMESPACE}}", namespace)

    with open(f"{output_path}/yamls/DeploymentNginxGw.yaml", "w") as file:
        file.write(f)

    print("Deployment Created!")
