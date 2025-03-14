#LIBTORCH := /usr/local/libtorch

ifeq ($(LIBTORCH),)
	LIBTORCH := $(abspath ../py/venv/lib/python3.12/site-packages/torch)
	# PyTorch libs are compiled with the old ABI
	ABI:=-D_GLIBCXX_USE_CXX11_ABI=0
else
	ABI:=
endif

export LD_LIBRARY_PATH:=/usr/local/cuda:$(LIBTORCH)/lib:$(LD_LIBRARY_PATH):$(PWD)

CFLAGS=-I$(LIBTORCH)/include/ -I$(LIBTORCH)/include/torch/csrc/api/include/ -O3
CPPFLAGS=-std=c++17 $(CFLAGS) $(ABI) -fPIC
LDFLAGS=-L/usr/local/cuda/lib64 -L$(LIBTORCH)/lib -ltorch -pthread -ltorch_cpu -lc10 -Wl,-rpath,$(LIBTORCH)/lib,-rpath,$(abspath .)

main: libtest_torch.so
	GO_CFLAGS="$(CFLAGS)" GO_CPPFLAGS="$(CPPFLAGS)" GO_LDFLAGS="$(LDFLAGS)" go build main.go
	@echo export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)

libtest_torch.so: test_torch.o
	$(CXX) $(CPPFLAGS) -std=c++17 -O3 -shared -o $(@) $(^) $(LDFLAGS)

clean:
	rm -f main libtest_torch.so test_torch.o
