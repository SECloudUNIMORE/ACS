#!/bin/bash

set -ue

NARGS=2

help () {
  >&2 echo "$(basename "$0") <output-dir> <input-trace-1> [<input-trace-2> ... ]"
  exit 2
}

test $# -ge "${NARGS}" || help

OUTPUT_DIR="$(realpath "${1}")"

shift 1

sourve venv/bin/activate

./scripts/create-sparse-batch.sh "${OUTPUT_DIR}" "$@"

mkdir -v "${OUTPUT_DIR}/cfiles"

./scripts/sparse-to-c-batch.sh byod.py 1 "${OUTPUT_DIR}/cfiles" "${OUTPUT_DIR}/0.5"/*

./python/byod.py trace-to-c -T "${OUTPUT_DIR}/cfiles/trace.h" -t "${OUTPUT_DIR}/trace" -bs 1 -l 50

mkdir -p "${OUTPUT_DIR}/timings"

./scripts/timings-batch.sh ./c-code/sparse_speed 100000 "${OUTPUT_DIR}/trace" "${OUTPUT_DIR}/timings" "${OUTPUT_DIR}/0.5"/*
