
import torch
import numpy as np
import pandas as pd
import sklearn

print("=" * 40)
print("       DEEPSHIELD ENVIRONMENT")
print("=" * 40)

print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)

print("\nCPU available:", torch.backends.cpu.get_cpu_capability())
print("DeepShield environment is ready!")
