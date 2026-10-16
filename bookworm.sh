#!/bin/sh
# Run Bookworm with the project's .venv, ignoring any PYTHONPATH from other envs (e.g. conda/ROS).
# Usage: ./bookworm.sh --config config/phone_mount.yaml ui
unset PYTHONPATH
exec "$(dirname "$0")/.venv/bin/python" -m bookworm "$@"
