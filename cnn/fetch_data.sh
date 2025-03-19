#!/bin/bash

_curl() {
    curl -L -C - --output-dir "${2-.}" --output $(basename "${1}") "${1}"
}

_wget() {
    wget -P "${2-.}" -c "${1}"
}

_httpget() {
    if which -s -- curl; then
        echo _curl
    elif which -s -- wget; then
        echo _wget
    else
        echo 'return 1;'
    fi
}

main() {
    if md5sum --quiet -c ./data/checksum; then
        return
    fi
    local -r cmd="$(_httpget)"
    for i in t10k train; do
        "${cmd}" "https://raw.githubusercontent.com/sunsided/mnist/refs/heads/master/${i}-images-idx3-ubyte.gz" data
        "${cmd}" "https://raw.githubusercontent.com/sunsided/mnist/refs/heads/master/${i}-labels-idx1-ubyte.gz" data
    done
    wait
    for i in data/*.gz; do
        gunzip --force -- "$i"
    done
}

main
