from stab import Grader, np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True, linewidth=10000)

names = ["XXX", "ZZZ"]
dxs = [0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001]
ldxs = np.log10(dxs)

results = {}
code_lengths = {}

for name in names:
    G = Grader(name, "Pauli", y=0.01)
    lenc = len(G.codes)
    code_lengths[name] = lenc
    code_results = []

    for ci in range(lenc):
        grads = []
        for dx in dxs:
            gradient = G.grad(ci, dx)
            grads.append(gradient)
        code_results.append(grads)

    results[name] = np.array(code_results)

maxci = max(len(v) for v in results.values())
fig, axes = plt.subplots(maxci, 1, figsize=(7, 3 * maxci), sharex=True)

for ci in range(maxci):
    ax = axes[ci]
    for name in names:
        ax.plot(ldxs, results[name][ci], marker="o", label=name)

    ax.set_title(f"$|{ci}_L\\rangle$")
    ax.set_ylabel("$\\mathcal{S} = \\nabla f$")
    ax.set_ylim(0.85, 1)
    ax.xaxis.set_inverted(True)
    ax.grid(True, ls="--", alpha=0.33)
    ax.legend()

axes[-1].set_xlabel("$\\Delta x$")
plt.tight_layout()
plt.savefig("sanity.png", dpi=300)
plt.show()
