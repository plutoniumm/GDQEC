import matplotlib.pyplot as plt
from stab import Grader, np

np.set_printoptions(precision=4, suppress=True, linewidth=10000)

# names = ["513", "D5"]
names = ["D5", "513"]
markers = ["^", "v"]
dx = 0.0001

yrange = np.arange(0.001, 0.034, 0.001)

print(f"dx = {dx}")
for name in names:
    gset = []
    for y in yrange:
        G = Grader(name, "Pauli", y=float(y))
        grads = G.sens(dx)
        print(f"{y}: ", grads)
        gset.append(grads)
    # endfor
    gset = np.array(gset).T

    for i in range(len(gset)):
        plt.plot(
            yrange,
            gset[i],
            label=f"{name} $|{i}_L\\rangle$",
            marker=markers[i],
            linestyle="dotted",
        )

    plt.xlabel("$\\gamma$")
    plt.ylabel("$\\mathcal{S}$")
# endfor

plt.legend()
plt.grid()
plt.show()
plt.savefig("images/d5v513.png", dpi=300)
