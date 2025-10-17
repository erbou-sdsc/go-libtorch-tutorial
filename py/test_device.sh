main() {
    if [[ ! -d ./venv || ! -e ./venv/bin/activate ]]; then
        python3.13 -m venv venv || return 1
        . ./venv/bin/activate
        pip3 install --upgrade pip
        pip3 install torch numpy
    elif  [[ ! -e ./venv/bin/activate ]]; then
	    echo "venv is not usable"
	    return 1
    else
        . ./venv/bin/activate || return 1
    fi
    ./test_device.py
    deactivate
}

main
