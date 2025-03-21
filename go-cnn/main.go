package main

/*
#cgo LDFLAGS: -L. -ltest_cnn
#include "test_cnn.h"
*/
import "C"

import (
    "unsafe"

    "fmt"
    "math/rand"
    "sync"
    "time"
)

// Go Bindings to C functions in test_cnn.cpp
// {
func InitializeModel() *C.struct_CNN {
    return C.torch_initialize_model()
}

func DeleteModel(model *C.struct_CNN) {
    C.torch_delete_model(model)
}

func TrainModel(model *C.struct_CNN, data []float32, target []int, num_epochs int) {
    cData := (*C.float)(unsafe.Pointer(&data[0]))
    cTarget := (*C.int)(unsafe.Pointer(&target[0]))
    cSize := (C.size_t)(len(data))
    cEpochs := (C.int)(num_epochs)
    C.torch_training(model, cData, cTarget, cSize, cEpochs)
}

func InferenceModel(model *C.struct_CNN, data []float32, result []float32) int {
    cData := (*C.float)(unsafe.Pointer(&data[0]))
    cSize := (C.size_t)(len(data))
    cResult := (*C.float)(unsafe.Pointer(&result[0]))
    cMaxSize := (C.size_t)(len(result))
    return int(C.torch_inference(model, cData, cSize, cResult, cMaxSize))
}
// }

type InferenceInput struct {
    Data []float32
    Callback chan<- []float32
}

func InferenceAggregator(model *C.struct_CNN, aggregatorChannel <-chan InferenceInput, batchSize int) {
    outputSize   := int(C.torch_model_output_size(model))
    inputSize    := int(C.torch_model_input_size(model))
    batchData    := make([]float32, 0, batchSize * inputSize)
    batchChan    := make([]chan<- []float32, 0, batchSize)
    resultBuffer := make([]float32, batchSize * outputSize)

    for true {
        timeout := false
        select {
            case request := <- aggregatorChannel:
                batchData = append(batchData, request.Data...)
                batchChan = append(batchChan, request.Callback)
            case <-time.After(1 * time.Second):
	        timeout = true
                fmt.Printf(`O`)
        }
        if (len(batchChan) >= batchSize || timeout) && len(batchData) > 0 {
            fmt.Printf(`*`)
            resultSize := InferenceModel(model, batchData, resultBuffer)
            fmt.Printf(`+`)
            if resultSize > 0 {
                for i, callback := range batchChan {
                    callback <- resultBuffer[i*outputSize : (i+1)*outputSize]
                }
            }
            batchData = batchData[:0]
            batchChan = batchChan[:0]
        }
    }
}

func simGenerateRandomData(id int, aggregatorChannel chan<- InferenceInput, dataSize int, wg *sync.WaitGroup) {
    callback := make(chan []float32, 1)
    defer close(callback)
    defer wg.Done()

    for range 4 {
        data := make([]float32, dataSize)
        for i := 0; i < dataSize; i++ {
            data[i] = rand.Float32()
        }
        fmt.Print(`:`)
        aggregatorChannel <- InferenceInput{ data, callback }
        select {
            case result := <- callback:
                if (len(result) == 0) {
                    fmt.Printf(`(%v)`, id)
                } else {
                    fmt.Print(`|`)
                }
            case <-time.After(2 * time.Second):
                fmt.Print(`?`)
        }
    }
    fmt.Print(`.`)
}

func main() {
    model := InitializeModel()
    defer DeleteModel(model)

    inputSize := int(C.torch_model_input_size(model))

    num := 100
    fmt.Printf("Request aggregator, batch size %v x {%v}, receives from %v random data generator goroutines\n\n", num, inputSize, 2*num)

    data := make([]float32, num * inputSize)
    target := make([]int, num)
    TrainModel(model, data, target, 2)

    aggregatorChannel := make(chan InferenceInput, num)
    defer close(aggregatorChannel)

    go InferenceAggregator(model, aggregatorChannel, num)

    var wg sync.WaitGroup
    wg.Add(2*num)

    fmt.Println(`>>>>>>>>>> codes <<<<<<<<<<`)
    fmt.Println(`:   A goroutine sent a request to request aggregator`)
    fmt.Println(`*   Request aggregator sent a batch of requests for inference`)
    fmt.Println(`+   Request aggregator received a batch of results and dispatched them to goroutines`)
    fmt.Println(`|   A goroutine received a result`)
    fmt.Println(`O   Request aggregator timed out while waiting for new input (submit partial batch)`)
    fmt.Println(`?   A goroutine timed out while waiting for the result`)
    fmt.Println(`.   A goroutine terminated`)
    fmt.Println(`$   End`)
    fmt.Print("\n\n")

    for k := range 2*num {
        go simGenerateRandomData(k, aggregatorChannel, inputSize, &wg)
    }

    wg.Wait()
    fmt.Println(`$`)
}
