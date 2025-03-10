main() {
    if [[ ! -d ./venv ]]; then
        python3 -m venv venv
        . ./venv/bin/activate
        pip3 install --upgrade pip
        pip3 install torch numpy
    else
        . ./venv/bin/activate
    fi
    ./mps.py
    deactivate
}

main
