### Simple C++ Torch tutorial with Go binding

#### Pre-requisites

This has been tested on MacOS arm64 (MPS) Sequoia, with command line xcode installed (make, clang* etc).

You must first extract the **LibTorch** C++ zip from https://pytorch.org/ in this folder.

#### Verify your hardware configuration

The code in this tutorial is configured to use an MPS device on MacOS arm64 architectures.
You can verify that you have MPS by executing the following code (this will create a venv with pytorch).

```
cd py
. ./test_mps.sh
```

#### Verify the c++ compilation environment

Before attempting the Go/C++ binding make sure that you are at least able to compile this simple C++ example.

```
cd ./c++
make test_torch
./test_torch
```

The LDFLAGS of makefile should work on MacOS arm64 sequoia with xcode command line tools installed. The same computer should be used for building and testing the executable.
Otherwise, on Linux and MacOs, you'll need to set _LD_LIBRARY_PATH_ or _DYLD_LIBRARY_PATH_ on MacOS to include the path to the libtorch lib folder or make the content of this folder libraries globally available (or modify -rpath in the makefile). On Windows you are on your own.

#### Test the Go/C++ binding

```
cd ./go
make
export DYLD_LIBRARY_PATH=${PWD}:${PWD}/../libtorch/lib:${DYLD_LIBRARY_PATH} # See comment below
./main
```

The makefile is configured to compile on MacOS Sequoia, arm64 (GOARCH=arm64).

Same comment for C++ above: the same computer must be used for compiling and building the executable. Furthermore the main will work only if executed from within its directory  (i.e. `./main`). Otherwise, you need to export the _LD_LIBRARY_PATH_ or _DYLD_LIBRARY_PATH_ environment variable to include both the libtest_sum.so folder _and_ the libtorch lib folder as demonstrated in the example.
