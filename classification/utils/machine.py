import torch
from multiprocessing import cpu_count

workers = max(cpu_count() - 1, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

