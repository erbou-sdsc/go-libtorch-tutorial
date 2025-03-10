package main

/*
#include "test_sum.h"
#cgo LDFLAGS: -L. -L../libtorch/lib -ltest_sum -Wl,-rpath ../libtorch/lib
*/
import "C"

import (
	"fmt"
	"unsafe"
)

func main() {
	goFloats := []float32{1.1, 2.2, 3.3, 4.4, 5.5}

	cFloats := (*C.float)(unsafe.Pointer(&goFloats[0]))

	x := C.test_sum(C.int(len(goFloats)), cFloats)
        fmt.Println(x)
}
