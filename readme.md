## sensitivity

Getting the gradient vector for each code

f=0.9155

```py
dx = 0.0001

print(f"dx = {dx}")
gset = []
G = Grader("513", "Pauli", y=0.01)

lenc = len(G.codes)
grads = []
for ci in range(lenc):
    vec = G.grad_vec(ci, dx)
    grads.append(vec)
    print(f"ci={ci}, grad={vec}")

```