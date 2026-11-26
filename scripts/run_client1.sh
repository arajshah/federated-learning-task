#!/usr/bin/env bash
# Start Flower client 1 with its configured dataset (configs/client1.yaml).
set -euo pipefail

python -m src.client.fl_client --config configs/client1.yaml
