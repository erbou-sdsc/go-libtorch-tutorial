#LIBTORCH := /usr/local/libtorch

ifeq ($(LIBTORCH),)
	LIBTORCH := $(abspath ../py/venv/lib/python3.12/site-packages/torch)
	# PyTorch libs are compiled with the old ABI
	ABI:=-D_GLIBCXX_USE_CXX11_ABI=0
else
	ABI:=
endif

CFLAGS=-I$(LIBTORCH)/include/ -I$(LIBTORCH)/include/torch/csrc/api/include/ -O3
CPPFLAGS=-std=c++17 $(CFLAGS) $(ABI) -fPIC
LDFLAGS=-L$(LIBTORCH)/lib -ltorch -pthread -ltorch_cpu -lc10 -Wl,-rpath $(LIBTORCH)/lib

main: libtest_sum.so
	GO_CFLAGS="$(CFLAGS)" go build main.go
	@echo set LD_LIBRARY_PATH=$(LIBTORCH)/lib:$${LD_LIBRARY_PATH}:$(abspath .)

libtest_sum.so: test_sum.o
	$(CXX) $(CPPFLAGS) -std=c++17 -O3 -shared -o $(@) $(^) $(LDFLAGS)

clean:
	rm -f main libtest_sum.so test_sum.o
