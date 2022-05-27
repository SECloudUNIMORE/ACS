#!/bin/bash

set -ue

help () {
	>&2 echo "$(basename "$0") <rle-bitsize>"
	exit 2
}

test $# -ge 1 || help
RLE_BITSIZE=$1

MAIN=sparse_speed
MAIN32="${MAIN}_32"

CFILES="src/types.c misc/files.c src/byod.c misc/timings.c"
DEFINES="-Dcompressed=1 -Duse_tables=1 -Dblock_type_size=1 -Drle_bitsize=${RLE_BITSIZE}"
WARNINGS="-Wall -Wformat -Wconversions"
CFLAGS="-O9 "
gcc ${CFLAGS} -o "${MAIN}" "src_cli/${MAIN}.c" ${CFILES} ${DEFINES} -lm
gcc ${CFLAGS} -o "${MAIN}_32" "src_cli/${MAIN}.c" ${CFILES} ${DEFINES} -lm -m32

