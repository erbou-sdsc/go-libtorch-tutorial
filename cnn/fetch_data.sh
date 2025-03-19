#!/usr/bin/bash

main() {
    mkdir -p data
    for i in t10k train; do
        wget -P data -c "https://raw.githubusercontent.com/sunsided/mnist/refs/heads/master/${i}-images-idx3-ubyte.gz" &
        wget -P data -c "https://raw.githubusercontent.com/sunsided/mnist/refs/heads/master/${i}-labels-idx1-ubyte.gz" &
    done
    wait
    for i in data/*.gz; do
        gunzip -- "$i"
    done
}

main
