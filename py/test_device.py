#!/usr/bin/env python3

import torch

# Check which supported device is available

if not torch.cuda.is_available():
    if not torch.cuda.is_built():
        print("The current PyTorch install was not built with cuda enabled.")
else:
    device = torch.device("cuda")
    print(f"Device {device} is enabled.")

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("The current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
else:
    device = torch.device("mps")
    print(f"Device {device} is enabled.")

x = torch.ones(5, device=device)
y = x * 2
print(f"Device {device} installed and working")
