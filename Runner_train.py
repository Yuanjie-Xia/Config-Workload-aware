import subprocess
import time
import requests
import os


class Runner:
    def __init__(self, jmeter_file_address, grafana_api_url, grafana_api_key):
        self.jmeter_file_address = jmeter_file_address
        self.grafana_api_url = grafana_api_url  # Example: http://<grafana_host>:3000/api/ds/query
        self.grafana_api_key = grafana_api_key  # Grafana API key with read access

    def run_jmeter_file(self, id, w_extra_stop=False, throughput=0):
        import os, time, subprocess, signal

        print("Starting JMeter test...")
        out_csv = "trainticket/jmeter_output.csv"
        out_jtl = "trainticket/results.jtl"

        # Clean previous outputs
        for p in (out_csv, out_jtl):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        if not w_extra_stop:
            try:
                subprocess.run(
                    [
                        "jmeter",
                        "-n",
                        "-t", self.jmeter_file_address,
                        "-l", out_jtl,
                        "-Jjmeter.save.saveservice.response_data=true",
                        "-Jjmeter.save.saveservice.autoflush=true",
                        "-j", "trainticket/jmeter.log",
                    ],
                    check=True
                )
                print("JMeter execution finished!")
                if os.path.exists(out_csv):
                    os.rename(out_csv, f"trainticket/jmeter_output_{id}.csv")
                if os.path.exists(out_jtl):
                    os.rename(out_jtl, f"trainticket/results_{id}.jtl")
            except subprocess.CalledProcessError as e:
                print(f"Error running JMeter: {e}")
            return

        # ------------------ Watchdog mode ------------------
        CHECK_INTERVAL_SEC = 180  # 3 minutes

        # Start JMeter in its own process group/session (so we can kill the whole tree)
        popen_kwargs = {
            "stdout": None,
            "stderr": None,
            "bufsize": 1,
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        else:  # Windows
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(
            [
                "jmeter",
                "-n",
                "-t", self.jmeter_file_address,
                "-l", out_jtl,
                "-Jjmeter.save.saveservice.response_data=true",
                "-Jjmeter.save.saveservice.autoflush=true",
                "-j", "trainticket/jmeter.log",
            ],
            **popen_kwargs
        )

        def _count_data_rows(path):
            """Fast line count; subtract 1 for header if present."""
            if not os.path.exists(path):
                return 0
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return max(0, sum(1 for _ in f) - 1)
            except Exception:
                return 0

        def _rows_and_src():
            # Prefer your CSV if it exists; otherwise fall back to JTL
            if os.path.exists(out_csv):
                return _count_data_rows(out_csv), "CSV"
            return _count_data_rows(out_jtl), "JTL"

        def _stop_proc_tree(p, reason=""):
            print(f"[WATCHDOG] stopping JMeter ({reason}) ...", flush=True)
            try:
                if os.name == "posix":
                    # graceful SIGTERM to the whole group
                    os.killpg(p.pid, signal.SIGTERM)
                else:
                    try:
                        p.send_signal(signal.CTRL_BREAK_EVENT)
                    except Exception:
                        pass
                    p.terminate()

                try:
                    p.wait(timeout=10)
                    return
                except Exception:
                    pass

                # escalate to hard kill
                print("[WATCHDOG] escalation to hard kill...", flush=True)
                if os.name == "posix":
                    os.killpg(p.pid, signal.SIGKILL)
                else:
                    subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    p.wait(timeout=5)
                except Exception:
                    pass
            except Exception as e:
                print(f"[WATCHDOG] stop error: {e}", flush=True)

        # wait up to 30s for an output file to appear
        start_wait = time.time()
        while not (os.path.exists(out_csv) or os.path.exists(out_jtl)):
            if proc.poll() is not None:
                break
            if time.time() - start_wait > 30:
                print("[WATCHDOG] no output file after 30s; continuing anyway...", flush=True)
                break
            time.sleep(0.25)

        last_rows, src = _rows_and_src()
        print(f"[WATCHDOG] monitoring {src}; baseline={last_rows}; threshold={throughput * 3} per 3min", flush=True)

        last_check = time.time()
        try:
            while True:
                # exit once JMeter ends
                if proc.poll() is not None:
                    break

                now = time.time()
                if now - last_check >= CHECK_INTERVAL_SEC:
                    last_check = now
                    current_rows, src = _rows_and_src()
                    delta = current_rows - last_rows
                    last_rows = current_rows
                    print(f"[WATCHDOG] 3-min delta={delta} (src={src}), threshold={throughput * 3}", flush=True)

                    if throughput > 0 and delta < throughput * 3:
                        _stop_proc_tree(proc, reason=f"delta {delta} < {throughput * 3}")
                        break

                time.sleep(0.2)

        finally:
            if proc.poll() is None:
                _stop_proc_tree(proc, reason="finalize")

            print("JMeter execution finished!", flush=True)
            try:
                if os.path.exists(out_csv):
                    os.rename(out_csv, f"trainticket/jmeter_output_{id}.csv")
            except Exception as e:
                print(f"[WARN] rename CSV failed: {e}")
            try:
                if os.path.exists(out_jtl):
                    os.rename(out_jtl, f"trainticket/results_{id}.jtl")
            except Exception as e:
                print(f"[WARN] rename JTL failed: {e}")

    def fetch_grafana_metrics(self):
        print("Fetching monitoring data from Grafana...")

        headers = {
            "Authorization": f"Bearer {self.grafana_api_key}",
            "Content-Type": "application/json"
        }

        # Example Grafana query (adapt depending on your datasource & query language)
        payload = {
            "queries": [
                {
                    "refId": "A",
                    "intervalMs": 1000,
                    "maxDataPoints": 1000,
                    "datasource": {"type": "prometheus", "uid": "your-prometheus-uid"},
                    "expr": 'rate(container_cpu_usage_seconds_total[1m])'
                }
            ],
            "range": {
                "from": f"{int(time.time() - 600)}000",  # Last 10 minutes
                "to": f"{int(time.time())}000"
            }
        }

        response = requests.post(
            self.grafana_api_url,
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            data = response.json()
            print("Grafana monitoring data retrieved successfully:")
            print(data)
        else:
            print(f"Failed to fetch Grafana data: {response.status_code}")
            print(response.text)
