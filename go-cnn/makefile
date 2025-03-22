#LIBTORCH := /usr/local/libtorch

ifeq ($(LIBTORCH),)
	LIBTORCH := $(abspath $(wildcard ../py/venv/lib/python3*/site-packages/torch))
	# PyTorch libs are compiled with the old ABI
	ABI:=-D_GLIBCXX_USE_CXX11_ABI=0
else
	ABI:=
endif

CFLAGS=-I$(LIBTORCH)/include/ -I$(LIBTORCH)/include/torch/csrc/api/include/ -O3
CPPFLAGS=-std=c++17 $(CFLAGS) $(ABI) -fPIC
LDFLAGS=-L/usr/local/cuda/lib64 -L$(LIBTORCH)/lib -ltorch -pthread -ltorch_cpu -lc10 -Wl,-rpath,$(LIBTORCH)/lib,-rpath,$(abspath .)

main: libtest_cnn.so main.go
	GO_CFLAGS="$(CFLAGS)" GO_CPPFLAGS="$(CPPFLAGS)" GO_LDFLAGS="$(LDFLAGS)" go build main.go
	@echo export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)

lib%.so: %.o
	$(CXX) $(CPPFLAGS) -std=c++17 -O3 -shared -o $(@) $(^) $(LDFLAGS)

clean:
	rm -f main libtest_cnn.dylib libtest_cnn.so test_cnn.o
