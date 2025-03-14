### Simple C++ Torch tutorial with Go binding

#### Pre-requisites

##### MacOS MPS

The MacOS code has been tested on MacOS arm64 (MPS) Sequoia, with command line xcode installed (make, clang* etc).

**Link against libtorch++**

Extract the appropriate **LibTorch** C++ zip from https://pytorch.org/ to a folder and export LIBTORCH to libtorch inside that folder.

**Link against pytorch's libraries**

Execute the `./test_device.sh` command, or run `pip install torch` and export LIBTORCH to lib/python3.12/site-packages/torch inside your python environment.
Note that the pytorch libraries have been compiled with the older C++ ABI, and you will need to compile with _-D_GLIBCXX_USE_CXX11_ABI=0_.

##### Linux CUDA

This has been tested on Linux Ubuntu 24.04 with CUDA 12.2 and Nvidia A100.

You need _g++, build-essential,_ and _libxml2_ 

Verify your cuda driver and cuda toolkit version (_sudo apt install pciutils_ if you don't have _lspci_):

```
lspci | grep -i nvidia
nvidia-smi
nvcc --version
```

If _nvcc_ is not found, install the version of cuda tooklkit from [nvidia developer](https://developer.nvidia.com/cuda-downloads/).

#### Verify your hardware configuration

The code in this tutorial is configured to use an MPS device on MacOS arm64 or CUDA on linux amd64
You can verify that you have a supported device by executing the following code (this will create a venv with pytorch, with the old C++ ABI).

```
cd py
. ./test_device.sh
```

#### Verify the c++ compilation environment

Before attempting the Go/C++ binding make sure that you are at least able to compile this simple C++ example.

```
cd ./c++
make test_torch
./test_torch
```

If the executable fails because it cannot find a library you must to set _LD_LIBRARY_PATH_ or _DYLD_LIBRARY_PATH_ on MacOS
to include the path to the libtorch lib folder or make the content of this folder libraries globally available. On Windows you are on your own.

#### Test the Go/C++ binding

```
cd ./go
make
export DYLD_LIBRARY_PATH=${PWD}:${PWD}/../libtorch/lib:${DYLD_LIBRARY_PATH} # See comment below
./main
```

The makefile is configured to compile on MacOS Sequoia, arm64 (GOARCH=arm64).

Same comment for C++ above. If it cannot find a library you must to export the _LD_LIBRARY_PATH_ or _DYLD_LIBRARY_PATH_ environment
variable to include both the libtest_sum.so folder _and_ the libtorch lib folder as demonstrated in the example.
