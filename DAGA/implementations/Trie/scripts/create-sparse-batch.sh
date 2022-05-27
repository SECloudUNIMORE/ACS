#!/usr/bin/env bash

set -eu

help () {
  >&2 echo "$(basename "$0") <output-dir> <input-trace-1> [<input-trace-2> ... ]"
  exit 2
}


directory_exists () {
  >&2 echo "Directory $1 already exists"
  exit 2
}

FFORMAT=_format
FC=_counters
FV=_values
FS=_skips

create_model () {
  WINDOW_SIZE="$1"
  SKIP_FACTOR="$2"
  TARGET_DIR="${OUTPUT_DIR}/${SKIP_FACTOR}/${WINDOW_SIZE}"
  mkdir -v -p -- "${TARGET_DIR}"
  cd -- "${TARGET_DIR}"
  "${CMD}" create-model-sparse "${OUTPUT_DIR}/trace" -n 50 -w "${WINDOW_SIZE}" -V "${FV}" -C "${FC}" -F "${FFORMAT}" -bs 1 -S "${FS}" -rs 2 -sf "${SKIP_FACTOR}"
  cd -
}


test $# -ge 2 || help
OUTPUT_DIR=$(realpath "$1")
test -f "${OUTPUT_DIR}" && directory_exists "${OUTPUT_DIR}"

shift 1

#CUR_PATH=$(realpath "$(pwd)")
#CMD="${CUR_PATH}/byod.py"
CMD=byod.py

mkdir -v "${OUTPUT_DIR}"

"${CMD}" trace-encode "$@" -O "${OUTPUT_DIR}/trace" -E "${OUTPUT_DIR}/encoding" -bs 1

for i in `seq 2 12`; do
  for j in 0.5; do
    create_model "${i}" "${j}"
  done
done
