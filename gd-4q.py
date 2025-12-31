from stab import Grader, np, Code
from torch.optim import Adam
import numpy.linalg as LA
import torch as pt
import os
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True, linewidth=10000)

F32, C64 = pt.float32, np.complex64
toNP = lambda x: x.detach().cpu().numpy()

steps, lr = 100, 0.0001

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)
history = {"loss": [], "fid": [], "norm": [], "ortho": []}

G = Grader("leung", "AD", y=0.2)
codes = G.codes.copy()
np.random.seed(0)
codes += (np.random.rand(*codes.shape) - 0.5) * 0.1

lenc, lenw = len(codes), len(codes[0])
cre = pt.tensor(codes.real, dtype=F32, requires_grad=True)
cim = pt.tensor(codes.imag, dtype=F32, requires_grad=True)

opt = Adam([cre, cim], lr=lr)


def grade(grads):
    cre.grad = pt.zeros_like(cre)
    cim.grad = pt.zeros_like(cim)
    cre.grad[ci] = pt.tensor(grads[:lenw], dtype=F32)
    cim.grad[ci] = pt.tensor(grads[lenw:], dtype=F32)


def getOrtho(c0, c1):
    ll = len(c0)
    ortho = c0.dot(c1.conj())
    f, g = ortho.real, ortho.imag
    # 0L
    # dOrtho/dx_i = 2f u_i - 2g v_i
    # dOrtho/dy_i = 2f v_i + 2g u_i
    # 1L
    # dOrtho/du_i = 2f x_p + 2g y_p
    # dOrtho/dv_i = 2f y_p - 2g x_p

    grads_0 = np.zeros(2 * ll)
    grads_1 = np.zeros(2 * ll)
    for p in range(ll):
        x_p, y_p = c0[p].real, c0[p].imag
        u_p, v_p = c1[p].real, c1[p].imag

        dK_dx = 2 * f * u_p - 2 * g * v_p
        dK_dy = 2 * f * v_p + 2 * g * u_p

        grads_0[p] = dK_dx
        grads_0[p + ll] = dK_dy

        dK_du = 2 * f * x_p + 2 * g * y_p
        dK_dv = 2 * f * y_p - 2 * g * x_p

        grads_1[p] = dK_du
        grads_1[p + ll] = dK_dv

    return np.array([grads_0, grads_1])


for step in range(steps):
    G.codes = (toNP(cre) + 1j * toNP(cim)).astype(C64)
    G.base = G.get_fid(G.codes)
    α, β = 2, 2

    L_orth = getOrtho(*G.codes)
    for ci in range(lenc):
        X, Y = G.codes[ci].real, G.codes[ci].imag
        L_n_X = 4 * β * (LA.norm(G.codes[ci]) - 1) * X
        L_n_Y = 4 * β * (LA.norm(G.codes[ci]) - 1) * Y

        L_norm = np.concatenate([L_n_X, L_n_Y])
        L_fidel = G.split_gvec(ci, dx=1e-4) * (2 * G.base - 1)
        L_ortho = L_orth[ci] * α

        grade(L_fidel + L_ortho + L_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)

        G.base = G.get_fid(G.codes)

    norm = np.sum((1 - LA.norm(G.codes, axis=1)) ** 2)
    ortho = np.abs(G.codes[0].dot(G.codes[1].conj()))
    fid = G.get_fid(G.codes)

    loss = (1.0 - fid) ** 2 + ortho + norm
    print(f"{step+1}/{steps}: f={fid:.4f}, O={ortho:.1e}, N={norm:.1e}, L={loss:.1e}")

    # Log metrics
    history["fid"].append(float(fid))
    history["loss"].append(float(loss))
    history["norm"].append(float(norm))
    history["ortho"].append(float(ortho))

# Save metrics
_steps = np.arange(1, len(history["loss"]) + 1)
data = {
    "steps": _steps,
    "loss": np.array(history["loss"]),
    "fid": np.array(history["fid"]),
    "norm": np.array(history["norm"]),
    "ortho": np.array(history["ortho"]),
    "code": G.codes,
}
np.savez(os.path.join(OUT_DIR, "gd513_decayed.npz"), **data)

# Plot metrics
fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
axes[0].plot(_steps, history["loss"], label="Loss", color="tab:red")
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(_steps, history["fid"], label="Fidelity", color="tab:blue")
axes[1].plot(_steps, history["norm"], label="Norm penalty", color="tab:green")
axes[1].plot(_steps, history["ortho"], label="Ortho penalty", color="tab:purple")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Metric")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gd513_decayed.png"), dpi=150)
plt.close(fig)
