import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local Flower smoke test with one server and two clients."
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--skip-data-generation", action="store_true")
    return parser.parse_args()


def wait_for_port(host: str, port: int, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"Server did not open {host}:{port} within {timeout_seconds}s")


def start_process(name: str, args: list[str], env: dict[str, str]) -> subprocess.Popen[str]:
    print(f"Starting {name}: {' '.join(args)}")
    return subprocess.Popen(
        [sys.executable, "-u", *args],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def collect_output(name: str, process: subprocess.Popen[str], timeout: int) -> str:
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stop_process(process)
        output, _ = process.communicate(timeout=5)
        raise TimeoutError(f"{name} timed out after {timeout}s\n{output}") from exc
    return output


def ensure_data(env: dict[str, str], skip_data_generation: bool) -> None:
    if skip_data_generation:
        return
    subprocess.run(
        [sys.executable, "scripts/generate_synthetic_data.py", "--overwrite"],
        cwd=ROOT,
        env=env,
        check=True,
    )


def verify_outputs() -> None:
    metrics_path = ROOT / "outputs/server_metrics.csv"
    final_parameters_path = ROOT / "outputs/final_global_parameters.npz"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not final_parameters_path.exists():
        raise FileNotFoundError(f"Missing final parameters file: {final_parameters_path}")

    metrics_lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    round_rows = max(0, len(metrics_lines) - 1)
    if round_rows < 3:
        raise RuntimeError(f"Expected at least 3 metric rows, found {round_rows}")

    print(f"Verified {round_rows} metric rows in {metrics_path}")
    print(f"Verified final parameters at {final_parameters_path}")


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(ROOT)

    ensure_data(env, args.skip_data_generation)
    for path in [
        ROOT / "outputs/server_metrics.csv",
        ROOT / "outputs/final_global_parameters.npz",
    ]:
        path.unlink(missing_ok=True)

    processes: dict[str, subprocess.Popen[str]] = {}
    try:
        processes["server"] = start_process(
            "server",
            ["-m", "src.server.fl_server", "--config", "configs/server.yaml"],
            env,
        )
        wait_for_port("127.0.0.1", 8080, timeout_seconds=30)

        processes["client1"] = start_process(
            "client1",
            ["-m", "src.client.fl_client", "--config", "configs/client1.yaml"],
            env,
        )
        processes["client2"] = start_process(
            "client2",
            ["-m", "src.client.fl_client", "--config", "configs/client2.yaml"],
            env,
        )

        server_output = collect_output("server", processes["server"], args.timeout)
        client1_output = collect_output("client1", processes["client1"], 30)
        client2_output = collect_output("client2", processes["client2"], 30)

        outputs = {
            "server": server_output,
            "client1": client1_output,
            "client2": client2_output,
        }
        for name, process in processes.items():
            if process.returncode != 0:
                raise RuntimeError(
                    f"{name} exited with code {process.returncode}\n{outputs[name]}"
                )

        verify_outputs()
        print("Local Flower smoke test passed.")
    finally:
        for process in processes.values():
            stop_process(process)


if __name__ == "__main__":
    main()
