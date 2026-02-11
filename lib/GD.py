from qudit.noise import Recovery as R
from qudit.tools import Fidelity as F
import numpy.linalg as LA
from cache import Cache
from .kraus import Krauser
from .codes import Code
import numpy as np


def Recov(rec: str):
    assert hasattr(R, rec)
    group = rec == "dutta"

    if rec == "petz":
        RF = R.petz
    elif rec == "cafaro":
        RF = R.cafaro
    elif rec == "leung":
        RF = R.leung
    else:  # "dutta"
        RF = R.dutta

    return group, RF


class Grader:
    def __init__(self, name: str, style: str, y: float, rec: str = "petz"):
        assert 0 <= y <= 1

        self.ckey = f"{name}-{style}-{y}-{rec}"
        self.name = name
        codes = Code(name)
        self.codes = codes.astype(np.complex64)

        Q = int(np.log2(len(codes[0])))

        group, RF = Recov(rec)

        if style == "AD":
            Ek = Krauser.AD(Q, 3, y, False)
        elif style == "Pauli_X":
            Ek = Krauser.Pauli(Q, 3, p=y, paulis=["X"])
        elif style == "Pauli_Y":
            Ek = Krauser.Pauli(Q, 3, p=y, paulis=["Y"])
        elif style == "Pauli_Z":
            Ek = Krauser.Pauli(Q, 3, p=y, paulis=["Z"])
        elif style == "Pauli":
            Ek = Krauser.Pauli(Q, 2, p=y, paulis=["X", "Y", "Z"], group=group)

        if style == "AD":
            Ak = Krauser.AD_full(Q, y)
        elif style.startswith("Pauli"):
            Ak = Krauser.Pauli_full(Q, p=y, paulis=["X", "Y", "Z"])

        self.Ek, self.Ak = Ek, Ak
        self.Rks = RF(Ek, codes)
        self.base = self.get_fid(codes)

    def get_fid(self, c) -> float:
        Rks, Ak = self.Rks, self.Ak
        return F.entanglement(Rks, Ak, c)

    # |c>[i] -> |c>[i] + e^iθ
    # cost = |f(C+) - f(C)| + |f(C-) - f(C)|/2
    def diff(self, theta, c, i, dx, codes=None):
        if codes is None:
            codes = self.codes

        delta = dx * np.exp(1j * theta)
        A = codes[c]

        A[i] += delta
        pert = self.get_fid(codes)
        A[i] -= delta

        slope = (pert - self.base) / dx

        return slope

    def wirt(self, c, i, dx, codes=None):
        if codes is None:
            codes = self.codes

        delx = self.diff(0, c, i, dx, codes)
        dely = self.diff(np.pi / 2, c, i, dx, codes)

        return np.array([delx, dely])

    def split_gvec(self, ci, dx):
        lenw = len(self.codes[0])
        grads = np.zeros(2 * lenw)

        for i in range(lenw):
            delx, dely = self.wirt(ci, i, dx=dx)

            grads[i] = delx
            grads[i + lenw] = dely

        return -grads

    def grad_vec(self, ci, dx, codes=None):
        if codes is None:
            codes = self.codes

        lenw = len(codes[0])
        grads = np.zeros(lenw)
        cache = Cache(str(dx))

        for i in range(lenw):
            if cache.has(self.ckey, ci, i):
                w = cache.get(self.ckey, ci, i)
            else:
                w = self.wirt(ci, i, dx=dx, codes=codes)

                cache.set(self.ckey, ci, i, w)
            grads[i] = LA.norm(w)

        return grads

    def grad(self, ci, dx, codes):
        return LA.norm(self.grad_vec(ci, dx, codes))

    def sens(self, dx, codes=None):
        if codes is None:
            codes = self.codes

        lenc = len(self.codes)
        grads = np.zeros(lenc)
        for ci in range(lenc):
            grads[ci] = self.grad(ci, dx, codes)

        return grads
