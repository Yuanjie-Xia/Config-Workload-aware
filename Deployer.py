import json
import subprocess

from kubernetes import client, config
from kubernetes.client.rest import ApiException


class Deployer:
    def __init__(self, service_name_set, work_model_json, namespace="default"):
        self.service_names_set = service_name_set  # ['s1', 's2', 's3']
        self.work_model = work_model_json
        self.configmap = None
        self.namespace = namespace

    def create_workmodel_configmap_data(self):
        data_dict = dict()
        metadata = client.V1ObjectMeta(
            name="workmodel",
            namespace=self.namespace
        )
        data_dict["workmodel.json"] = json.dumps(self.work_model)
        self.configmap = client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            data=data_dict,
            metadata=metadata
        )

    def deploy_configmap(self):
        print("######################")
        print(f"We are going to DEPLOY the the configmap: {self.configmap.metadata.name}")
        print("######################")

        config.load_kube_config()
        api_instance = client.CoreV1Api()
        try:
            api_response = api_instance.create_namespaced_config_map(
                namespace=self.namespace,
                body=self.configmap
            )
            print(f"ConfigMap '{self.configmap.metadata.name}' created.")
            print("---")
        except ApiException as e:
            print("Exception when calling CoreV1Api->create_namespaced_config_map: %s\n" % e)

    def undeploy_configmap(self):
        print("######################")
        print(f"We are going to UNDEPLOY the the configmap: {self.configmap.metadata.name}")
        print("######################")
        config.load_kube_config()
        api_instance = client.CoreV1Api()
        try:
            api_response = api_instance.delete_namespaced_config_map(
                namespace=self.namespace,
                name=self.configmap.metadata.name)
            print(f"ConfigMap '{self.configmap.metadata.name}' deleted.")
            print("---")
        except ApiException as e:
            print("Exception when calling CoreV1Api->delete_namespaced_config_map: %s\n" % e)

    def restart_deployments(self):
        for element in self.service_names_set:
            print(f"Restarting deployment: {element}")
            result = subprocess.run(
                ["kubectl", "rollout", "restart", f"deployment/{element}", "-n", self.namespace],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                print(f"[ERROR] Failed to restart {element}: {result.stderr}")
                continue
            else:
                print(f"[OK] Restarted {element}")

            # --- Wait until rollout is complete ---
            print(f"Waiting for rollout of {element} to complete...")
            status = subprocess.run(
                ["kubectl", "rollout", "status", f"deployment/{element}", "-n", self.namespace, "--timeout=300s"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if status.returncode != 0:
                print(f"[ERROR] Rollout for {element} failed or timed out: {status.stderr}")
            else:
                print(f"[OK] Deployment {element} successfully rolled out.")

    def update_dns_setting(self):
        for deployment in self.service_names_set:
            print(f"Processing deployment: {deployment}")

            # Get pod(s) for the deployment (assuming label app=<deployment>)
            pod_cmd = [
                "kubectl", "get", "pods",
                "-l", f"app={deployment}",
                "-o", "json",
                "-n", f"{self.namespace}"
            ]
            result = subprocess.run(pod_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if result.returncode != 0:
                print(f"[ERROR] Failed to get pods for {deployment}: {result.stderr}")
                continue

            pods = json.loads(result.stdout).get("items", [])
            if not pods:
                print(f"[WARN] No pods found for deployment {deployment}")
                continue

            node_name = pods[0]["spec"].get("nodeName", "")
            print(f"[INFO] {deployment} is running on node: {node_name}")

            # Determine patch file based on node type
            if "fse" in node_name:
                patch_file = "dns-patch-fse.yaml"
            elif "ase" in node_name:
                patch_file = "dns-patch-ase.yaml"
            else:
                print(f"[WARN] Unknown node type for {deployment} (node: {node_name}) — skipping DNS patch")
                continue

            # Apply the patch
            patch_cmd = [
                "kubectl", "patch", "deployment", deployment,
                "-n", self.namespace,
                "--type", "merge",
                "--patch-file", patch_file
            ]
            patch_result = subprocess.run(patch_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if patch_result.returncode != 0:
                print(f"[ERROR] Failed to patch {deployment} with {patch_file}: {patch_result.stderr}")
            else:
                print(f"[OK] Patched {deployment} with {patch_file}")

            rollout_cmd = [
                "kubectl", "rollout", "status", f"deployment/{deployment}",
                "-n", self.namespace,
                f"--timeout=30s"
            ]
            rollout_result = subprocess.run(rollout_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if rollout_result.returncode != 0:
                print(f"[ERROR] Deployment {deployment} failed to roll out: {rollout_result.stderr}")
            else:
                print(f"[OK] Deployment {deployment} successfully rolled out")

