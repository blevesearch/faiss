#!/bin/bash
set -e

SRC_DIR=$1
INSTALL_DIR=$2

configure_args="--prefix=${INSTALL_DIR} --with-install-suffix=_faiss"

# Profiling only supported on non-Darwin.
if [ $(uname -s) != "Darwin" ]; then
    configure_args+=" --enable-prof"
fi

# Change directory to the source directory
cd "${SRC_DIR}"

./autogen.sh ${configure_args} >/dev/null
make -j8 build_lib_shared >/dev/null
make install_lib_shared install_include install_bin >/dev/null