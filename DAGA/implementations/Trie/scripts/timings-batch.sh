#! /bin/bash

set -ue


help () {
  >&2 echo "$(basename "$0") <cmd> <ntests> <trace> <output-dir> <input-dir-1> [<input-dir-2> ... ]"
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

test $# -ge 5 || help
CMD=$1
NTESTS=$2
TRACE=$3
OUTPUT_DIR=$(realpath "$4")
test -f "${OUTPUT_DIR}" && directory_exists "${OUTPUT_DIR}"

shift 4

for DIR in "$@"; do
  OUT="${OUTPUT_DIR}/$(basename ${DIR})"
  echo "Computing timings for ${DIR}"
  taskset 1 "${CMD}" "${TRACE}" "${NTESTS}" "${DIR}"/{${FFORMAT},${FC},${FV},${FS}} > "${OUT}"
done
