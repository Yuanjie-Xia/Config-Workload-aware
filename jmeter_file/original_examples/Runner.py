from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, Future
import sched
import time
import threading
from TimingError import TimingError
import requests
import json
import sys
import os
import shutil
from typing import Any, List, Dict, Tuple, Callable
import importlib
from pprint import pprint
from functools import partial


import QueryStringBuilder as qsb

import argparse
import argcomplete


class Counter(object):
    def __init__(self, start=0):
        self.lock = threading.Lock()
        self.value = start

    def increase(self):
        self.lock.acquire()
        try:
            self.value = self.value + 1
        finally:
            self.lock.release()

    def decrease(self):
        self.lock.acquire()
        try:
            self.value = self.value - 1
        finally:
            self.lock.release()


def do_requests(
    event,
    stats,
    local_latency_stats,
    query_builder: qsb.HeaderFactory,
    endpoint_picker: callable,
):
    global processed_requests, last_print_time_ms, error_requests, pending_requests
    # pprint(workload[event]["services"])
    # for services in event["services"]:
    # print(services)
    processed_requests.increase()
    try:
        now_ms = time.time_ns() // 1_000_000
        if runner_type in {"greedy", "timely_greedy"}:
            pending_requests.increase()

        headers = query_builder.build_headers()
        endpoint = endpoint_picker(event=event, headers=headers)
        req_url = f"{ms_access_gateway}/{endpoint}"
        r = requests.get(req_url, headers=headers)
        pending_requests.decrease()

        if r.status_code != 200:
            print("Response Status Code", r.status_code)
            error_requests.increase()

        req_latency_ms = int(r.elapsed.total_seconds() * 1000)
        req_stats = [
            now_ms,
            req_latency_ms,
            r.status_code,
            processed_requests.value,
            pending_requests.value,
        ]
        req_stats = list([str(e) for e in req_stats])
        req_stats.extend([f'"{key}:{value}"' for key, value in headers.items()])
        stats_output = " \t ".join(req_stats)
        stats.append(stats_output)
        local_latency_stats.append(req_latency_ms)

        if now_ms > last_print_time_ms + 5_000:
            print(
                f"Processed request {processed_requests.value}, latency {req_latency_ms}, pending requests {pending_requests.value}"
            )
            last_print_time_ms = now_ms
        return event["time"], req_latency_ms
    except Exception as err:
        print("Error: %s" % err)
        raise err


def job_assignment(
    v_pool: ThreadPoolExecutor,
    v_futures: list,
    event,
    stats,
    local_latency_stats,
    query_builder: qsb.HeaderFactory,
    endpoint_picker: callable,
    on_complete_callback: "callable | None",
):
    global timing_error_requests, pending_requests
    try:
        worker = v_pool.submit(
            do_requests,
            event,
            stats,
            local_latency_stats,
            query_builder,
            endpoint_picker,
        )
        if on_complete_callback:
            worker.add_done_callback(on_complete_callback)
        v_futures.append(worker)
        if runner_type not in {"greedy", "timely_greedy"}:
            pending_requests.increase()
        if pending_requests.value > threads:
            # maximum capacity of thread pool reached, request is queued (not an issue for greedy runner)
            if runner_type not in {"greedy", "timely_greedy"}:
                timing_error_requests += 1
                raise TimingError(event["time"])
    except TimingError as err:
        print("Error: %s" % err)
        raise err


def file_runner(workload=None):
    global start_time, stats, local_latency_stats, header_builder

    stats = list()
    print("###############################################")
    print("############   Run Forrest Run!!   ############")
    print("###############################################")
    if len(sys.argv) > 1 and workload is None:
        workload_file = sys.argv[1]
    else:
        workload_file = workload

    with open(workload_file) as f:
        workload = json.load(f)
    s = sched.scheduler(time.time, time.sleep)
    pool = ThreadPoolExecutor(threads)
    futures = list()

    for event in workload:
        # in seconds
        # s.enter(event["time"], 1, job_assignment, argument=(pool, futures, event))
        # in milliseconds
        s.enter(
            (event["time"] / 1000 + 2),
            1,
            job_assignment,
            argument=(pool, futures, event, stats, local_latency_stats, header_builder),
        )

    start_time = time.time()
    print("Start Time:", datetime.now().strftime("%H:%M:%S.%f - %g/%m/%Y"))
    s.run()

    wait(futures)
    run_duration_sec = time.time() - start_time
    avg_latency = 1.0 * sum(local_latency_stats) / len(local_latency_stats)

    print("###############################################")
    print("###########   Stop Forrest Stop!!   ###########")
    print("###############################################")
    print(
        "Run Duration (sec): %.6f" % run_duration_sec,
        "Total Requests: %d - Error Request: %d - Timing Error Requests: %d - Average Latency (ms): %.6f - Request rate (req/sec) %.6f"
        % (
            len(workload),
            error_requests.value,
            timing_error_requests,
            avg_latency,
            1.0 * len(workload) / run_duration_sec,
        ),
    )

    if run_after_workload is not None:
        args = {
            "run_duration_sec": run_duration_sec,
            "last_print_time_ms": last_print_time_ms,
            "requests_processed": processed_requests.value,
            "timing_error_number": timing_error_requests,
            "total_request": len(workload),
            "error_request": error_requests.value,
            "runner_results_file": f"{output_path}/{result_file}_{workload_var.split('/')[-1].split('.')[0]}.txt",
        }
        run_after_workload(args)


def select_endpoint_simple(event: Dict[str, Any], *args, **kwargs) -> str:
    return event["service"]


def select_endpoint_by_header(
    event: Dict[str, Any], headers: Dict, header_key: str
) -> str:
    endpoint = event["service"][headers[header_key]]
    return endpoint


runner_start_time: datetime = None
timely_runner_is_done: bool = False
last_processed_message: int = -1


def get_endpoint_picker(runner_params: dict) -> Tuple[Callable, str]:
    picker = None
    srv = None

    if "ingress_service" in runner_params.keys():
        picker = select_endpoint_simple
        srv = runner_params["ingress_service"]
    else:
        picker = select_endpoint_simple
        srv = "s0"

    if isinstance(srv, dict):
        picker = partial(select_endpoint_by_header, header_key=srv["header_key"])
        srv = runner_params["ingress_service"]["services"]

    return picker, srv


def timely_greedy_runner():
    """A greedy runner that runs for some amount of time."""
    global start_time, stats, local_latency_stats, runner_parameters, header_builder, endpoint_picker, runner_start_time

    print(f"{runner_parameters=}")

    stats = list()
    print("###############################################")
    print("############   Run Forrest Run!!   ############")
    print("###############################################")

    s = sched.scheduler(time.time, time.sleep)
    pool = ThreadPoolExecutor(threads)
    futures: list[Future] = list()
    endpoint_picker, srv = get_endpoint_picker(runner_parameters)
    print(f"{srv=}, {endpoint_picker=}")
    event = {"service": srv, "time": 0}
    slow_start_end = 32  # number requests with initial delays
    slow_start_delay = 0.1

    def on_response_received(_fut: Future):
        global runner_start_time, timely_runner_is_done, last_processed_message
        runner_current_time = datetime.now()
        runner_passed_time = runner_current_time - runner_start_time
        runner_minutes_spent = runner_passed_time.seconds / 60.0
        if (
            not timely_runner_is_done
            and runner_minutes_spent > max_runner_time_in_minutes
        ):
            timely_runner_is_done = True
            # Kills all of the remaining requests.
            print(f"Passed the maximum time: {max_runner_time_in_minutes} minutes.")
            pool.shutdown(wait=False)
            for idx, future in enumerate(futures):
                if not future.done():
                    future.cancel()
                    if last_processed_message == -1:
                        last_processed_message = idx

    try:
        runner_start_time = datetime.now()
        # put every request in the thread pool scheduled at time 0 (in case with initial slow start spread to reduce initial concurrency)
        # HACK: the timely runner needs a large workload definition to function; this shouldn't be necessary.
        for i in range(workload_events):
            if i < slow_start_end:
                event_time = i * slow_start_delay
            s.enter(
                delay=event_time,
                priority=1,
                action=job_assignment,
                argument=(
                    pool,
                    futures,
                    event,
                    stats,
                    local_latency_stats,
                    header_builder,
                    endpoint_picker,
                    on_response_received,
                ),
            )

        start_time = time.time()
        print("Start Time:", datetime.now().strftime("%H:%M:%S.%f - %g/%m/%Y"))

        s.run()

        wait(futures)

    except KeyboardInterrupt:
        pool.shutdown(wait=True)
        for future in futures:
            if not future.done():
                future.cancel()
        raise

    run_duration_sec = time.time() - start_time
    avg_latency = 1.0 * sum(local_latency_stats) / len(local_latency_stats)

    if not timely_runner_is_done:
        run_duration_min = run_duration_sec / 60.0
        print(
            f"WARNING: The timely runner ran out of requests, it only ran for {run_duration_min:.1f}/{max_runner_time_in_minutes} minutes!"
        )
        better_workload_events_value = int(
            workload_events * (max_runner_time_in_minutes / run_duration_min)
        )
        print(
            f"It's recommended to update `workload_events` to at least {better_workload_events_value}."
        )

    print("###############################################")
    print("###########   Stop Forrest Stop!!   ###########")
    print("###############################################")

    print(
        "Run Duration (sec): %.6f" % run_duration_sec,
        "Total Requests: %d (approx.) - Error Request: %d - Timing Error Requests: %d - Average Latency (ms): %.6f - Request rate (req/sec) %.6f (approx.)"
        % (
            last_processed_message,
            error_requests.value,
            timing_error_requests,
            avg_latency,
            1.0 * last_processed_message / run_duration_sec,
        ),
    )

    if run_after_workload is not None:
        args = {
            "run_duration_sec": run_duration_sec,
            "last_print_time_ms": last_print_time_ms,
            "requests_processed": processed_requests,
            "timing_error_number": timing_error_requests,
            "total_request": workload_events,
            "error_request": error_requests,
            "runner_results_file": f"{output_path}/{result_file}.txt",
        }
        run_after_workload(args)


def greedy_runner():
    global start_time, stats, local_latency_stats, runner_parameters, header_builder, endpoint_picker

    print(f"{runner_parameters=}")

    endpoint_picker = None
    if "ingress_service" in runner_parameters.keys():
        endpoint_picker = select_endpoint_simple
        srv = runner_parameters["ingress_service"]
    else:
        endpoint_picker = select_endpoint_simple
        srv = "s0"

    if isinstance(srv, dict):
        endpoint_picker = partial(
            select_endpoint_by_header, header_key=srv["header_key"]
        )

    stats = list()
    print("###############################################")
    print("############   Run Forrest Run!!   ############")
    print("###############################################")

    s = sched.scheduler(time.time, time.sleep)
    pool = ThreadPoolExecutor(threads)
    futures: list[Future] = list()
    event = {"service": srv, "time": 0}
    slow_start_end = 32  # number requests with initial delays
    slow_start_delay = 0.1
    try:
        # put every request in the thread pool scheduled at time 0 (in case with initial slow start spread to reduce initial concurrency)
        for i in range(workload_events):
            if i < slow_start_end:
                event_time = i * slow_start_delay
            s.enter(
                delay=event_time,
                priority=1,
                action=job_assignment,
                argument=(
                    pool,
                    futures,
                    event,
                    stats,
                    local_latency_stats,
                    header_builder,
                    endpoint_picker,
                ),
            )

        start_time = time.time()
        print("Start Time:", datetime.now().strftime("%H:%M:%S.%f - %g/%m/%Y"))

        s.run()

        wait(futures)
    except KeyboardInterrupt:
        pool.shutdown(wait=True, cancel_futures=True)
        for future in futures:
            if not future.done():
                future.cancel()
        raise

    run_duration_sec = time.time() - start_time
    avg_latency = 1.0 * sum(local_latency_stats) / len(local_latency_stats)

    print("###############################################")
    print("###########   Stop Forrest Stop!!   ###########")
    print("###############################################")

    print(
        "Run Duration (sec): %.6f" % run_duration_sec,
        "Total Requests: %d - Error Request: %d - Timing Error Requests: %d - Average Latency (ms): %.6f - Request rate (req/sec) %.6f"
        % (
            workload_events,
            error_requests.value,
            timing_error_requests,
            avg_latency,
            1.0 * workload_events / run_duration_sec,
        ),
    )

    if run_after_workload is not None:
        args = {
            "run_duration_sec": run_duration_sec,
            "last_print_time_ms": last_print_time_ms,
            "requests_processed": processed_requests,
            "timing_error_number": timing_error_requests,
            "total_request": workload_events,
            "error_request": error_requests,
            "runner_results_file": f"{output_path}/{result_file}.txt",
        }
        run_after_workload(args)


def periodic_runner():
    global start_time, stats, local_latency_stats, runner_parameters, header_builder

    if "rate" in runner_parameters.keys():
        rate = runner_parameters["rate"]
    else:
        rate = 1

    if "ingress_service" in runner_parameters.keys():
        srv = runner_parameters["ingress_service"]
    else:
        srv = "s0"

    stats = list()
    print("###############################################")
    print("############   Run Forrest Run!!   ############")
    print("###############################################")

    s = sched.scheduler(time.time, time.sleep)
    pool = ThreadPoolExecutor(threads)
    futures = list()
    event = {"service": srv, "time": 0}
    offset = 10  # initial delay to allow the insertion of events in the event list
    for i in range(workload_events):
        event_time = offset + i * 1.0 / rate
        s.enter(
            event_time,
            1,
            job_assignment,
            argument=(pool, futures, event, stats, local_latency_stats, header_builder),
        )

    start_time = time.time()
    print("Start Time:", datetime.now().strftime("%H:%M:%S.%f - %g/%m/%Y"))
    s.run()

    wait(futures)
    run_duration_sec = time.time() - start_time
    avg_latency = 1.0 * sum(local_latency_stats) / len(local_latency_stats)

    print("###############################################")
    print("###########   Stop Forrest Stop!!   ###########")
    print("###############################################")

    print(
        "Run Duration (sec): %.6f" % run_duration_sec,
        "Total Requests: %d - Error Request: %d - Timing Error Requests: %d - Average Latency (ms): %.6f - Request rate (req/sec) %.6f"
        % (
            workload_events,
            error_requests.value,
            timing_error_requests,
            avg_latency,
            workload_events / run_duration_sec,
        ),
    )

    if run_after_workload is not None:
        args = {
            "run_duration_sec": run_duration_sec,
            "last_print_time_ms": last_print_time_ms,
            "requests_processed": processed_requests,
            "timing_error_number": timing_error_requests,
            "total_request": workload_events,
            "error_request": error_requests,
            "runner_results_file": f"{output_path}/{result_file}.txt",
        }
        run_after_workload(args)


### Main

RUNNER_PATH = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config-file",
    action="store",
    dest="parameters_file",
    help="The Runner Parameters file",
    default=f"{RUNNER_PATH}/RunnerParameters.json",
)


argcomplete.autocomplete(parser)

try:
    args = parser.parse_args()
except ImportError:
    print(
        "Import error, there are missing dependencies to install.  'apt-get install python3-argcomplete "
        "&& activate-global-python-argcomplete3' may solve"
    )
except AttributeError:
    parser.print_help()
except Exception as err:
    print("Error:", err)
    raise err

parameters_file_path = args.parameters_file

last_print_time_ms = 0
run_after_workload = None
header_builder = None
timing_error_requests = 0
processed_requests = Counter()
error_requests = Counter()
pending_requests = Counter()

try:
    with open(parameters_file_path) as f:
        params = json.load(f)
    runner_parameters = params["RunnerParameters"]
    header_builder = qsb.build_header_factory_from_runner_parameters(runner_parameters)
    endpoint_selector = None
    runner_type = runner_parameters["workload_type"]  # {workload (default), greedy}
    workload_events = runner_parameters["workload_events"]  # n. request for greedy
    max_runner_time_in_minutes = (
        int(runner_parameters["max_runner_time_in_minutes"])
        if "max_runner_time_in_minutes" in runner_parameters
        else 0
    )
    print(f"{max_runner_time_in_minutes=}")
    ms_access_gateway = runner_parameters[
        "ms_access_gateway"
    ]  # nginx access gateway_agg ip
    workloads = runner_parameters["workload_files_path_list"]
    threads = runner_parameters["thread_pool_size"]  # n. parallel threads
    round = runner_parameters["workload_rounds"]  # number of repetition rounds
    result_file = runner_parameters["result_file"]  # number of repetition rounds
    if "OutputPath" in params.keys() and len(params["OutputPath"]) > 0:
        output_path = params["OutputPath"]
        if output_path.endswith("/"):
            output_path = output_path[:-1]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    else:
        output_path = RUNNER_PATH
    if (
        "AfterWorkloadFunction" in params.keys()
        and len(params["AfterWorkloadFunction"]) > 0
    ):
        sys.path.append(params["AfterWorkloadFunction"]["file_path"])
        run_after_workload = getattr(
            importlib.import_module(
                params["AfterWorkloadFunction"]["file_path"].split("/")[-1]
            ),
            params["AfterWorkloadFunction"]["function_name"],
        )

except Exception as err:
    print("ERROR: in Runner Parameters,", err)
    raise err
    exit(1)


## Check if "workloads" is a directory path, if so take all the workload files inside it
if os.path.isdir(workloads[0]):
    dir_workloads = workloads[0]
    workloads = list()
    src_files = os.listdir(dir_workloads)
    for file_name in src_files:
        full_file_name = os.path.join(dir_workloads, file_name)
        if os.path.isfile(full_file_name):
            workloads.append(full_file_name)


stats = list()
local_latency_stats = list()
start_time = 0.0

if runner_type == "greedy":
    greedy_runner()
    with open(f"{output_path}/{result_file}.txt", "w") as f:
        f.writelines("\n".join(stats))

elif runner_type == "timely_greedy":
    timely_greedy_runner()
    with open(f"{output_path}/{result_file}.txt", "w") as f:
        f.writelines("\n".join(stats))

elif runner_type == "periodic":
    periodic_runner()
    with open(f"{output_path}/{result_file}.txt", "w") as f:
        f.writelines("\n".join(stats))
else:
    # default runner is "file" type
    for cnt, workload_var in enumerate(workloads):
        for x in range(round):
            print("Round: %d -- workload: %s" % (x + 1, workload_var))
            processed_requests.value = 0
            timing_error_requests = 0
            error_requests.value = 0
            file_runner(workload_var)
            print("***************************************")
        if cnt != len(workloads) - 1:
            print("Sleep for 100 sec to allow completion of previus requests")
            time.sleep(100)
        with open(
            f"{output_path}/{result_file}_{workload_var.split('/')[-1].split('.')[0]}.txt",
            "w",
        ) as f:
            f.writelines("\n".join(stats))
