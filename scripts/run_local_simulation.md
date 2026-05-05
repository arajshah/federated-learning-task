# Local Federated Simulation

This workflow runs one Flower FedAvg server and two local Flower clients on the same machine. Each client reads only its own configured CSV file.

From the repository root, install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate synthetic data for both clients if real data is not present:

```bash
python scripts/generate_synthetic_data.py --overwrite
```

## Three-Terminal Workflow

Terminal 1, start the server:

```bash
./scripts/run_server.sh
```

Terminal 2, start Client 1:

```bash
./scripts/run_client1.sh
```

Terminal 3, start Client 2:

```bash
./scripts/run_client2.sh
```

The server waits until both clients are available, runs the configured number of FedAvg rounds, writes aggregated metrics to `outputs/server_metrics.csv`, and saves final global parameter arrays to `outputs/final_global_parameters.npz`.

## Automated Smoke Test

To verify the same local setup from one command:

```bash
python scripts/run_fl_smoke_test.py
```

## Optional Local Baseline

You can still run standalone local training without Flower:

```bash
python -m src.client.train --config configs/client1.yaml
python -m src.client.train --config configs/client2.yaml
```
