#LIBTORCH := /usr/local/libtorch

ifeq ($(LIBTORCH),)
	LIBTORCH := $(abspath ../py/venv/lib/python3.12/site-packages/torch)
	# PyTorch libs are compiled with the old ABI
	ABI:=-D_GLIBCXX_USE_CXX11_ABI=0
else
	ABI:=
endif

CPPFLAGS=-std=c++17 -I$(LIBTORCH)/include/ -I$(LIBTORCH)/include/torch/csrc/api/include/ $(ABI)
LDFLAGS=-L$(LIBTORCH)/lib -ltorch -pthread -ltorch_cpu -lc10 -Wl,-rpath $(LIBTORCH)/lib

test_torch:

%: %.o
	$(CXX) $(@).o $(LDFLAGS) -o $(@)
	@echo set LD_LIBRARY_PATH=$(LIBTORCH)/lib:$${LD_LIBRARY_PATH}

test_torch.o:

clean:
	- rm -f test_torch test_torch.o
