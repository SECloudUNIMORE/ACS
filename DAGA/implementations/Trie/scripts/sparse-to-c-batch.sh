#!/bin/bash

set -ue

FFORMAT=_format
FC=_counters
FV=_values
FS=_skips

NARGS=4

help () {
  >&2 echo "$(basename "$0") <cmd> <block-size> <output-dir> <input-dir-1> [<input-dir-2> ... ]"
  exit 2
}

directory_exists () {
  >&2 echo "Directory $1 already exists"
  exit 2
}

test $# -ge "${NARGS}" || help

CMD="$(which $1)"
BS="$2"
OUTPUT_DIR="$(realpath "$3")"
test -f "${OUTPUT_DIR}" && directory_exists "${OUTPUT_DIR}"

shift 3

for DIR in "$@"; do
  T="$(basename ${DIR})"
  cd "${DIR}"
  OUT="${OUTPUT_DIR}/data${T}.h"
  echo "Encoding data structures to c file ${OUT}"
  "${CMD}" sparse-to-c -bs "${BS}" -f "${FFORMAT}" -v "${FV}" -c "${FC}" -s "${FS}" -out "${OUT}"
  cd -
done
