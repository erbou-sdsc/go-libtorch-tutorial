## Simple C++ Torch tutorial with Go binding

### Pre-requisites

#### Verify your hardware configuration

This tutorial is configured to use an MPS device on MacOS arm64 or CUDA on Linux amd64.
To verify that your system has a supported device, you can execute the following steps.
This process will create a virtual environment with PyTorch and the older C++ ABI in _py/venv_,
and it will verify your hardware configuration and confirm that the appropriate device (either MPS or CUDA) is available.

```
cd py
. ./test_device.sh
```

##### MacOS MPS

The MacOS code has been tested on MacOS arm64 (MPS) Sequoia, with command line xcode installed including make, clang v16.0.0.

##### Linux CUDA

This has been tested on Linux Ubuntu 24.04 with CUDA 12.2 and Nvidia A100.

You need _g++-12, build-essential,_ and _libxml2_. Note that CUDA Toolkit 12.2 will not work with a higher verison of g++ if using cmake.

Verify your CUDA driver and CUDA Toolkit version (_sudo apt install pciutils_ if you don't have _lspci_):

```
lspci | grep -i nvidia
nvidia-smi
nvcc --version
```

If _nvcc_ is not found, install the version of cuda tooklkit from [nvidia developer](https://developer.nvidia.com/cuda-downloads/).

#### LibTorch++

LibTorch is required in order to compile a C++ application.

Two options are possible:

##### Libtorch++

Extract the appropriate **LibTorch** C++ zip from https://pytorch.org/ to a folder, e.g. _/usr/local/libtorch_.

If you intend to use _make_ you should also _export LIBTORCH=/usr/local/libtorch_.

##### PyTorch's libraries

If you have already executed the _test_device.sh_ script, _libtorch_ will be installed
in the Python environment located at _py/venv_.

If you haven't run the script yet,  create the Python virtual environment and install PyTorch inside the virtual environment using pip:

```
python -m venv venv
. venv/bin/activate
pip install torch
```

If you intend to use _make_, you should also set the _LIBTORCH_ environment variable to the libtorch folder inside PyTorch:

```
    export LIBTORCH=$(python -c "import torch; print(torch.__path__[0])")/lib/python3.12/site-packages/torch
```

Note on C++ ABI Compatibility:
The version of PyTorch may have been compiled with the older C++ ABI, in which case you will get undefined symbols with
_cxx11_ during the compilation. If that happens, you must explicitly add the _-D_GLIBCXX_USE_CXX11_ABI=0_ flag to _CCFLAGS_ when building with make.
This is to ensures that the code is compiled using the same ABI as the PyTorch libraries.

### Verify the c++ compilation environment

Before attempting the Go/C++ binding make sure that you are at least able to compile this simple C++ example.

You have two options:

#### CMake

If you have _cmake_ installed, you can build the sample app as follows.

Set _CMAKE_PREFIX_PATH_ in _CMakeLists.txt_ to the home of the libtorch library.

```
cd c++
mkdir -p build
cd build
cmake ..
make
./test_torch
```

#### Make

CMake is excellent for portable builds, but troubleshooting can be difficult when issues arise.
To make things easier, we provide a simplified makefile that is easier to debug and adapt to your specific environments.

Set _LIBTORCH_ to the home folder of the libtorch library as indicated before.

```
cd c++
make test_torch
./test_torch
```

In both options, if the executable fails due to a missing library, set _LD_LIBRARY_PATH_ (Linux) or _DYLD_LIBRARY_PATH_ (MacOS)
to include the path to the libtorch lib folder.

### Test the Go/C++ binding

As before, you have too options.

#### CMake

Set _CMAKE_PREFIX_PATH_ in _CMakeLists.txt_ to the home folder of the libtorch library as appropriate.

```
cd go
mkdir -p build
cd build
cmake ..
make
cd ..
LD_LIBRARY_PATH=. ./main
```

#### Make

Set _LIBTORCH_ to the home folder of the libtorch library as indicated before.

```
cd go
make
LD_LIBRARY_PATH=. ./main
```

In both options, if the executable fails due to a missing library, set _LD_LIBRARY_PATH_ (Linux) or _DYLD_LIBRARY_PATH_ (MacOS)
to include the paths to the missing libraries.

