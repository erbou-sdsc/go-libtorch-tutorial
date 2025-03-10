all: libtest_sum.so
	GOARCH=arm64 go build main.go

libtest_sum.so: test_sum.cpp
	clang++ -std=c++17 -L../libtorch/lib -I../libtorch/include/ -I../libtorch/include/torch/csrc/api/include/ -O3 -arch arm64 -fPIC -shared -o libtest_sum.so test_sum.cpp -ltorch -lc10 -ltorch_cpu

clean:
	rm -f main libtest_sum.so test_sum.o
