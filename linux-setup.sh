#!/bin/bash


## Linux setup file
## It is meant to be run on a stock Ubuntu LTS 24.04, with nvidia A100 and CUDA 12.2 drivers.

GO_URL=${GO_URL-https://go.dev/dl/go1.24.1.linux-amd64.tar.gz}
LIBTORCH_URL=${LIBTORCH_URL-https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu124.zip}
CUDA_TKT_URL=${CUDA_TKT_URL-https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run}
LIBTORCH_HOME=/usr/local/libtorch

_install() {
    if [[ "${BASH_SOURCE}"x = x ]]; then
        local installer_dir=$(pwd -P)}
    else
	local installer_dir=$(dirname -- $(realpath -- "${BASH_SOURCE}"))
    fi
    local base_dir=$(dirname -- $(find "${installer_dir}" -name 'c++' -type d -print -quit))
    LIBTORCH_HOME=$(realpath ${LIBTORCH_HOME-.})

    if [[ $(uname -m) != "x86_64" ]]; then
        echo "Script not meant for arch $(uname -m)"
	return
    fi

    if [[ $(uname -s) != "Linux" ]]; then
        echo "Script not meant for $(uname -s) OS Kernel"
	return
    fi

    if ! command -v g++ || ! command -v wget || ! command -v unzip || ! command -v cmake; then
        export DEBIAN_FRONTEND=noninteractive
        apt update
        apt install -y wget g++-12 build-essential git libxml2 unzip vim tmux cmake python3.12 python3.12-venv
        apt -y autoremove
        for i in /bin/gcc*-12 /bin/g++*-12; do ln -fs $i ${i/-12/}; done
        ln -fs /bin/gcc /bin/cc
        ln -fs /bin/g++ /bin/c++
        ln -fs /usr/bin/python3.12 /usr/bin/python3
    fi

    nvidia-smi
    cat /etc/os-release|grep VERSION=
    uname -m

    export CUDA_HOME="/usr/local/cuda"
    export PATH="/usr/local/go/bin:/usr/local/cuda/bin${PATH+:${PATH}}"
    export LD_LIBRARY_PATH="${LIBTORCH_HOME-}/lib:/usr/local/cuda/lib:${CUDA_HOME}/lib:${base_dir}/go:${base_dir}/go/build${LD_LIBRARY_PATH+:${LD_LIBRARY_PATH}}"

    if ! command -v nvcc; then
	echo "Install CUDA toolkit"
        mkdir -p pkgs
        wget -P ./pkgs -c "${CUDA_TKT_URL}"
	chmod +x ./pkgs/$(basename "${CUDA_TKT_URL/\%2B/+}")
	./pkgs/$(basename "${CUDA_TKT_URL/\%2B/+}") --silent --toolkit
    fi

    if ! command -v go; then
	echo "Install golang to /usr/local/go"
        mkdir -p pkgs
        wget -P pkgs -c "${GO_URL}"
	tar -C /usr/local/ -xf ./pkgs/$(basename "${GO_URL}")
    fi

    if [[ ! -d "${LIBTORCH_HOME-x}" ]]; then
	echo "Install libtorch C++ to ${LIBTORCH_HOME}"
        mkdir -p pkgs
        wget -P pkgs -c "${LIBTORCH_URL}"
	unzip -q -d $(dirname "${LIBTORCH_HOME}") ./pkgs/$(basename "${LIBTORCH_URL/\%2B/+}")
    fi

    /usr/local/cuda/bin/nvcc --version

    echo "export CUDA_HOME=${CUDA_HOME}"
    echo "export LIBTORCH_HOME=${LIBTORCH_HOME}"
    echo "export PATH=${PATH}"
    echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
}

_install
