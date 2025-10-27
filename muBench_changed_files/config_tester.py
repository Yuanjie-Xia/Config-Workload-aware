import glob
import logging
import os
import subprocess
import shlex
import uuid
import random

import psutil
import time
import string

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def measure_subprocess_cpu_time(cmd):
    """Run a command and track subprocess CPU time while it is running."""
    try:
        process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ps_proc = psutil.Process(process.pid)

        cpu_time = 0.0
        wall_start = time.time()

        # Loop while the process is alive and update CPU usage
        while True:
            if process.poll() is not None:
                break
            try:
                times = ps_proc.cpu_times()
                cpu_time = times.user + times.system
                user_time = times.user
                system_time = times.system
            except psutil.NoSuchProcess:
                break
            time.sleep(0.1)  # sampling interval

        wall_end = time.time()

        stdout, stderr = process.communicate()

        logger.info(f"Command: {cmd}")
        logger.info(f"CPU Time: {cpu_time:.3f} sec, Wall Time: {wall_end - wall_start:.3f} sec")

        return {
            "success": process.returncode == 0,
            "cpu_time": cpu_time,
            "user_time": user_time,
            "system_time": system_time,
            "wall_start": wall_start,
            "wall_end": wall_end,
            "wall_time": wall_end - wall_start,
            "stdout": stdout.decode() if stdout else "",
            "stderr": stderr.decode() if stderr else "",
        }

    except Exception as e:
        logger.error(f"Failed to execute '{cmd}': {e}")
        return {
            "success": False,
            "cpu_time": None,
            "wall_time": None,
            "stdout": "",
            "stderr": str(e),
        }


def lrzip_config_tester(param):
    valid_algorithms = {'-z', '-b', '-g', '-n', '-l'}

    algorithm = param.get('algorithm')
    level = param.get('level')
    window = param.get('window')
    nice = param.get('nice')
    processor = param.get('processor')

    # Validate parameters
    if algorithm not in valid_algorithms:
        raise ValueError(f"Invalid algorithm: {algorithm}")
    if not (8 <= int(level) <= 9):
        raise ValueError(f"Level must be between 8 and 9, got {level}")
    if not (1 <= int(window) <= 99):
        raise ValueError(f"Window must be between 1 and 99, got {window}")
    if not (-20 <= int(nice) <= 19):
        raise ValueError(f"Nice must be between -20 and 19, got {nice}")
    if not (1 <= int(processor) <= 4):
        raise ValueError(f"Processor count must be between 1 and 4, got {processor}")

    # Find all chunk files
    chunk_files = glob.glob("/app/lrzip_chunks/chunk_*")

    if not chunk_files:
        return {"message": "No chunk files found in /app/lrzip_chunks", "cpu_time": 0.0}

    # Randomly pick one chunk
    # input_file = random.choice(chunk_files)
    input_file = chunk_files[0]
    output_file = f"/app/{os.path.basename(input_file)}-{uuid.uuid4().hex}.lrz"

    # Build command
    cmd = f"lrzip {algorithm} -L {level} -w {window} -N {nice} -p {processor} -o {output_file} {input_file}"
    result = measure_subprocess_cpu_time(cmd)

    if result["success"] and os.path.exists(output_file):
        logger.info(f"LRZIP CPU Time: {result['cpu_time']:.3f}s")
        os.remove(output_file)  # Optional cleanup
        return {"message": cmd, "cpu_time": result["cpu_time"], "wall_start": result["wall_start"],
                "wall_end": result["wall_end"], "wall_time": result["wall_time"], "user_time": result["user_time"], "system_time": result["system_time"]}
    else:
        return {"message": "LRZIP failed", "stderr": result["stderr"], "cpu_time": result["cpu_time"]}


def ffmpeg_config_tester(param):
    """
    Tests FFmpeg configuration by running an FFmpeg command with parameters from a dictionary.

    :param param: Dictionary containing FFmpeg parameters.
        Required keys:
            - input_file (str): Path to the input video file.
            - output_file (str): Path to the output video file.
        Optional keys:
            - codec (str): Video codec to use (default: libx264).
            - crf (int): Constant Rate Factor (default: 23, range: 0-51).
            - preset (str): Encoding speed preset (default: medium).
    :return: None
    """

    # Default values
    defaults = {
        "codec": "libx264",
        "crf": 23,
        "preset": "medium",
    }

    # Merge user-provided parameters with defaults
    config = {**defaults, **param}

    # Validate values
    valid_presets = ["ultrafast", "superfast", "faster", "fast", "medium", "slow", "slower"]
    if config["preset"] not in valid_presets:
        raise ValueError(f"Invalid preset '{config['preset']}'. Choose from {valid_presets}.")
    if not (0 <= int(config["crf"]) <= 51):
        raise ValueError(f"Invalid CRF value '{config['crf']}'. Must be between 0 and 51.")

    # Get a random chunk from the /app/chunks directory
    chunk_files = glob.glob("/app/chunks/blue_chunk_*.y4m")
    if not chunk_files:
        return {"message": "No video chunks found in /app/chunks", "cpu_time": 0.0}

    # input_file = random.choice(chunk_files)
    input_file = chunk_files[0]
    output_file = f"/app/output_video_{uuid.uuid4().hex}.mp4"

    # Construct FFmpeg command
    cmd = f"ffmpeg -i {shlex.quote(input_file)} -c:v {config['codec']} -crf {config['crf']} -preset {config['preset']} {shlex.quote(output_file)}"
    result = measure_subprocess_cpu_time(cmd)

    # Clean up and return result
    if result["success"]:
        logger.info(f"ffmpeg CPU Time: {result['cpu_time']:.3f}s")
        os.remove(output_file)  # Optional: remove output to save space
        return {"message": cmd, "cpu_time": result["cpu_time"], "wall_start": result["wall_start"],
                "wall_end": result["wall_end"], "wall_time": result["wall_time"], "user_time": result["user_time"], "system_time": result["system_time"]}
    else:
        return {"message": "ffmpeg failed", "stderr": result["stderr"], "cpu_time": result["cpu_time"]}


def sqlite_config_tester(param):
    # Default values for SQLite configuration options
    defaults = {
        "cache_size": "4000",
        "page_size": "1024",
        "autovacuum": True,
        "exclusive": True,
        "journal_mode": "WAL",
        "nosync": True,
        "stats": True,
        "output_file": f"/app/test_{uuid.uuid4().hex}.db"
    }

    # Merge defaults with passed parameters
    config = {**defaults, **param}

    # Build the command
    cmd = f"/sqlite/test/speedtest {config['output_file']}"
    cmd += f" --cachesize {config['cache_size']}"
    cmd += f" --pagesize {config['page_size']}"

    if config['autovacuum']:
        cmd += " --autovacuum"
    if config['exclusive']:
        cmd += " --exclusive"
    if config['journal_mode']:
        cmd += f" --journal {config['journal_mode']}"
    if config['nosync']:
        cmd += " --nosync"
    if config['stats']:
        cmd += " --stats"

    print(f"Executing command: {cmd}")
    result = measure_subprocess_cpu_time(cmd)

    # Clean up
    if os.path.exists(config['output_file']):
        os.remove(config['output_file'])
        print(f"Removed output file: {config['output_file']}")

    if result["success"]:
        logger.info(f"SQLITE CPU Time: {result['cpu_time']:.3f}s")
        return {
            "message": cmd,
            "cpu_time": result["cpu_time"],
            "wall_start": result["wall_start"],
            "wall_end": result["wall_end"],
            "wall_time": result["wall_time"]
        }
    else:
        return {
            "message": "SQLITE failed",
            "stderr": result["stderr"],
            "cpu_time": result["cpu_time"]
        }


def bandwidth_loader(params):
    # print("--------> Network stress start")
    # TODO: return the data of the made requests?
    mean_bandwidth = params['mean_bandwidth'] if 'mean_bandwidth' in params else 1
    bandwidth_load = random.expovariate(1 / mean_bandwidth)
    num_chars = int(max(1, 1000 * bandwidth_load))  # Response in kB
    response_body = "".join(
        random.choice(string.ascii_letters) for i in range(num_chars)
    )
    # print("--------> Network stress stop")
    return response_body


def config_tester(params):
    response_body = {}

    if "lrzip" in params and params["lrzip"]["run"]:
        response_body["lrzip"] = lrzip_config_tester(params["lrzip"])

    if "ffmpeg" in params and params["ffmpeg"]["run"]:
        response_body["ffmpeg"] = ffmpeg_config_tester(params["ffmpeg"])

    if "sqlite" in params and params["sqlite"]["run"]:
        response_body["sqlite"] = sqlite_config_tester(params["sqlite"])
    bandwidth_loader(params)

    return response_body


if __name__ == "__main__":
    config_tester({"ffmpeg": {
                        "run": True,
                        "input_file": "/app/akiyo_cif.y4m",
                        "output_file": "/app/output_video.mp4",
                        "codec": "libx264",
                        "crf": "21",
                        "preset": "veryslow"
                    }})