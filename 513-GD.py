from qudit.algo.statiliser import GramSchmidt
from stab import Grader, np, Code
from torch.optim import Adam
import torch as pt
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True, linewidth=10000)

F32, C64 = pt.float32, np.complex64
toNP = lambda x: x.detach().cpu().numpy()

steps, lr = 20, 5e-3

G = Grader("513", "Pauli", y=0.01)
codes = G.codes.copy()

lenc, lenw = len(codes), len(codes[0])
cre = pt.tensor(codes.real, dtype=F32, requires_grad=True)
cim = pt.tensor(codes.imag, dtype=F32, requires_grad=True)

opt = Adam([cre, cim], lr=lr)


def grade(grads):
    cre.grad = pt.zeros_like(cre)
    cim.grad = pt.zeros_like(cim)
    cre.grad[ci] = pt.tensor(grads[:lenw], dtype=F32)
    cim.grad[ci] = pt.tensor(grads[lenw:], dtype=F32)


fids = []

for step in range(steps):
    G.codes = (toNP(cre) + 1j * toNP(cim)).astype(C64)
    G.base = G.get_fid(G.codes)

    for ci in range(lenc):
        grade(G.split_gvec(ci, dx=1e-4))

        opt.step()
        opt.zero_grad(set_to_none=True)

        G.codes = toNP(GramSchmidt(cre + 1j * cim))
        if step >= 19:
            print(G.codes.round(4))

        G.base = G.get_fid(G.codes)

    norm = np.sum((1 - np.linalg.norm(G.codes, axis=1)) ** 2)
    ortho = np.abs(G.codes[0].dot(G.codes[1].conj()))
    fid = G.get_fid(G.codes)

    loss = 1.0 - fid + (ortho + norm) * 10
    print(f"{step+1}/{steps}: f={fid:.4f}, L={loss:.4f}")
    fids.append(float(fid))

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(fids) + 1), fids, marker="o")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss vs Steps")
plt.ylim(0.98, 1.0)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/gd_loss.png", dpi=300)
plt.show()
