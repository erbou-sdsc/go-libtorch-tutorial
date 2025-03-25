package main

// Display header of mnist data
// See also hexdump -v -C ./*-ubyte
// For payload: hexdump -s 16 -v -e '"%08_ax" 28/1 "%02x " "\n"' ./t10k-images-idx3-ubyte
// : skip 16 bytes header (4+4xdimensions), display 28x1-bytes per row

import (
    "encoding/binary"
    "fmt"
    "os"
    "log"
    "bytes"
)

type Header struct {
    Magic      uint16
    DataType   uint8
    NumDim     uint8
    Dim        [4]uint32
}

func main() {
    if len(os.Args) < 2 {
        fmt.Printf("Usage: %v mnist-data-ubyte\n", os.Args[0])
        os.Exit(1)
    }
    file, err := os.Open(os.Args[1])
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()

    headerSize := 20
    headerBytes := make([]byte, headerSize)

    _, err = file.Read(headerBytes)
    if err != nil {
        log.Fatal(err)
    }

    var header Header
    err = binary.Read(bytes.NewReader(headerBytes), binary.BigEndian, &header)
    if err != nil {
        log.Fatal(err)
    }

    // Print out the header fields
    fmt.Printf("Magic:    %d\n", header.Magic)
    fmt.Printf("DataType: %d\n", header.DataType)
    fmt.Printf("NumDim  : %d\n", header.NumDim)
    for i := uint8(0); i < header.NumDim; i++ {
        fmt.Printf("Dim%v    : %d\n", i, header.Dim[i])
    }
}
