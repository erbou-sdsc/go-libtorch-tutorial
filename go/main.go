package main

/*
#include "test_torch.h"
#cgo LDFLAGS: -L. -ltest_torch 
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
