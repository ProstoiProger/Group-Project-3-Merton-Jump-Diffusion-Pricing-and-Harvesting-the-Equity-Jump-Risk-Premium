import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

qqq = pd.read_csv("data/raw/qqq_daily.csv")

returns = pd.to_numeric(qqq["returns"], errors="coerce").dropna().to_numpy()

RV = np.sum(returns**2)
mu1 = np.sqrt(2 / np.pi)
BPV = (1 / mu1**2) * np.sum(np.abs(returns[:-1]) * np.abs(returns[1:]))

Z = np.sqrt(len(returns)) * ((RV / BPV) - 1)

print(f"Realized Variance (RV): {RV}")
print(f"Bipower Variation (BPV): {BPV}")
print(f"Jump Test Statistic Z: {Z}")