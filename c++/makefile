CPPFLAGS=-std=c++17 -I../libtorch/include/ -I../libtorch/include/torch/csrc/api/include/ -O3
LDFLAGS=-L../libtorch/lib/ -ltorch -pthread -lc10 -ltorch_cpu -Wl,-rpath $(CURDIR)/../libtorch/lib

test_torch:

clean:
	rm test_torch
