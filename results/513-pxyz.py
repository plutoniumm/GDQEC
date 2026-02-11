import sys
sys.path.append("../")

import matplotlib.pyplot as plt
from lib.GD import Grader, np

np.set_printoptions(precision=4, suppress=True, linewidth=10000)

channels = ["Pauli", "Pauli_X", "Pauli_Y", "Pauli_Z"]
markers = ["^", "v"]
dx = 0.0001

yrange = np.arange(0.001, 0.034, 0.001)

print(f"dx = {dx}")
for channel in channels:
    print(f"---{channel}---\n")
    gset = []
    for y in yrange:
        G = Grader("513", channel, y=float(y))
        grads = G.sens(dx)
        print(f"{channel} {y.round(4)}: ", grads)
        gset.append(grads)
    # endfor
    gset = np.array(gset).T

    if channel == "Pauli_X":
        gset += 0.01
    elif channel == "Pauli_Z":
        gset -= 0.01

    for i in range(len(gset)):
        plt.plot(
            yrange,
            gset[i],
            label=f"{channel} $|{i}_L\\rangle$",
            marker=markers[i % len(markers)],
            linestyle="-.",
        )

plt.xlabel("$\\gamma$")
plt.ylabel("$\\mathcal{S}$")
plt.legend()
plt.grid()
plt.savefig("images/pauli513.png", dpi=300)
plt.show()
