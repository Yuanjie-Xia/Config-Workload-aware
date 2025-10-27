from readline import append_history_file
import logging
import threading
import os
import glob
import random
import jsonmerge


LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

internal_service_function = None
internal_service_params_v = None

# Dinamyc import of all function in InternalServiceFunctions folder
for path in glob.glob("MSConfig/InternalServiceFunctions/[!_]*.py"):
    name, ext = os.path.splitext(os.path.basename(path))
    exec(f"from MSConfig.InternalServiceFunctions.{name} import *")


class InternalServiceExecutor(threading.Thread):
    def __init__(self, internal_service_function, params, response):
        threading.Thread.__init__(self)
        self.params = params
        self.internal_service_function = internal_service_function
        self.response = response

    def run(self):
        self.response.set_body(self.internal_service_function(self.params))
        # self.response.set_body(eval(self.internal_service_function))


class ThreadReturnedValue:
    def __init__(self):
        self.body = None

    def get_body(self):
        return self.body

    def set_body(self, body):
        self.body = body


def compute_pi(params):
    default_params = {"range_complexity": [50, 100], "mean_bandwidth": 10}
    params = jsonmerge.merge(default_params, params)

    cpu_load = random.randint(
        params["range_complexity"][0], params["range_complexity"][1]
    )
    pi_greco = list()

    q, r, t, k, m, x = 1, 0, 1, 1, 3, 3
    counter = 0
    while True:
        if 4 * q + r - t < m * t:
            # yield m
            pi_greco.append(m)
            q, r, t, k, m, x = (
                10 * q,
                10 * (r - m * t),
                t,
                k,
                (10 * (3 * q + r)) // t - 10 * m,
                x,
            )
            if counter > cpu_load - 1:
                break
            else:
                counter = counter + 1
        else:
            q, r, t, k, m, x = (
                q * k,
                (2 * q + r) * x,
                t * x,
                k + 1,
                (q * (7 * k + 2) + r * x) // (t * x),
                x + 2,
            )

    bandwidth_load = random.expovariate(1 / params["mean_bandwidth"])
    num_chars = max(1, 1000 * bandwidth_load)  # Response in kB
    response_body = "m" * int(num_chars)

    return response_body


def set_internal_service_function(internal_service_params):
    global internal_service_function, internal_service_params_v
    # HACK: this [0] refers to the "loader" field; it just picks the first key.
    function_name = (
        "loader"
        if len(internal_service_params) == 0
        else list(internal_service_params)[0]
    )
    internal_service_params_v = list(internal_service_params.values())[0]
    internal_service_function = eval(function_name)
    logger.info(
        'Setting internal service function: "{func}"'.format(func=function_name)
    )


def run_internal_service(internal_service_params):
    # global internal_service_function, internal_service_params_v
    # if internal_service_function == None:
    set_internal_service_function(internal_service_params)
    logger.info(internal_service_function)
    # function_name = list(internal_service_params)[0]
    # internal_service_params_v = list(internal_service_params.values())[0]
    # response = list()
    # response = dict()
    response = ThreadReturnedValue()
    thread = InternalServiceExecutor(
        internal_service_function, internal_service_params_v, response
    )
    thread.start()
    thread.join()
    return response.get_body()
