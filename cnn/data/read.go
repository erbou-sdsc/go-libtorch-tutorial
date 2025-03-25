package main

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
	Dim        [3]uint32
}

func main() {
	file, err := os.Open(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	headerSize := 128
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
