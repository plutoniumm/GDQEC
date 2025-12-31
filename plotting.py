import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
file = "gd513_decayed.npz"
path = os.path.join(OUT_DIR, file)

data = np.load(path, allow_pickle=True)
steps = data["steps"]
loss = np.abs(data["loss"])
fid = data["fid"]
norm = data["norm"]
ortho = data["ortho"]
code = data["code"].round(3)

i = 99
while fid[i] > 0.999:
    i -= 1
print(
    f"""Final:
Infidelity: {fid[i]:.5f}
Norm Loss: {norm[i]:.5f}
Perp Loss: {ortho[i]:5f}
"""
)

W = 10


def rolling_stat(x, w, func):
    n = len(x)
    out = np.empty(n, dtype=float)
    for i in range(n):
        s = max(0, i - w // 2)
        e = min(n, i + (w + 1) // 2)
        out[i] = func(x[s:e])
    return out


ma5 = rolling_stat(loss, W, np.mean)

fig, axes = plt.subplots(2, 1, figsize=(7, 9), sharex=True)

axes[0].plot(steps, loss, label="Loss", color="tab:red")
axes[0].plot(
    steps,
    ma5,
    label=f"Moving Average ({W})",
    color="black",
    linestyle=":",
    linewidth=1.5,
)
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(
    steps, np.abs(1.0 - fid), label=f"Infidelity: {1-fid[i]:.5f}", color="tab:blue"
)
axes[1].plot(steps, norm, label=f"Norm Loss: {norm[i]:.5f}", color="tab:green")
axes[1].plot(steps, ortho, label=f"Orth Loss: {ortho[i]:.5f}", color="tab:purple")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Metric")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gd_history_stats.png"), dpi=300)
plt.close(fig)
