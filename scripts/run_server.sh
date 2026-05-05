#!/usr/bin/env bash
set -euo pipefail

python -m src.server.fl_server --config configs/server.yaml
