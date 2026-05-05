# EMRAPT Federated Learning Prototype

This repository is a local prototype for federated learning on tabular radiomics data. It is designed to simulate a small federated setup on one machine with one server process and two client processes, where each client trains on its own local CSV file without sharing raw data.

The current phase runs a local Flower FedAvg prototype with one server process and two client processes. Each client trains against its own local CSV file and sends only model parameters and metrics to the server.

## Project Structure

```text
emrapt-fl/
  configs/          YAML configuration for the server and local clients
  data/             Local client data folders, not intended for raw data sharing
  src/common/       Shared config, data, metrics, model, and utility code
  src/client/       Client training and future Flower client entrypoints
  src/server/       Future Flower server entrypoint
  scripts/          Convenience scripts and local simulation notes
```

## Expected Data Format

Each client should keep its own CSV file under its local data directory, for example:

```text
data/client1/radiomics.csv
data/client2/radiomics.csv
```

CSV files are expected to contain:

- `ID`: sample identifier, dropped before training
- `Target`: binary label column, using values such as `0` and `1`
- Feature columns: numeric tabular radiomics features

Example:

```csv
ID,Target,shape_volume,firstorder_mean,glcm_contrast
P001,0,12.4,103.2,0.81
P002,1,18.9,120.5,1.14
```

Raw data should remain in each client's local folder. The federated workflow should exchange model parameters and metrics only.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Local Federated Simulation

Generate synthetic heterogeneous client data if real data is not present:

```bash
python scripts/generate_synthetic_data.py --overwrite
```

Start the server in Terminal 1:

```bash
./scripts/run_server.sh
```

Start Client 1 in Terminal 2:

```bash
./scripts/run_client1.sh
```

Start Client 2 in Terminal 3:

```bash
./scripts/run_client2.sh
```

The server waits for both clients, runs the configured FedAvg rounds, logs aggregated metrics to `outputs/server_metrics.csv`, and saves final global parameter arrays to `outputs/final_global_parameters.npz`.

To run the same flow as a single smoke test:

```bash
python scripts/run_fl_smoke_test.py
```

## Local Training Baseline

Standalone local training remains available for debugging each client before FL:

```bash
python -m src.client.train --config configs/client1.yaml
python -m src.client.train --config configs/client2.yaml
```

Training prints train loss, test accuracy, and test ROC-AUC after each epoch. Final checkpoints are saved under each client's configured `output_dir`.

## Configuration

Client configs define server address, dataset location, preprocessing, model, local training settings, and checkpoint output directory. Server config defines host, port, FedAvg client counts, round count, and output directory.
