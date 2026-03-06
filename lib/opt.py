from numpy.linalg import multi_dot as MD
from scipy.optimize import minimize
from qec import Fidelity, np

ctr = 0
class Resurrection:
    @staticmethod
    def parameterized(error_kraus: list, codes: list, weights: np.ndarray) -> list:
        global ctr
        print(f"{ctr}: {weights.sum().round(3)}", end="\r")
        ctr += 1
        Rks = []

        idx = 0
        for Eks in error_kraus:
            Rk = []
            for i in codes:
                w = weights[idx]
                idx += 1
                op = sum([np.outer(i, np.dot(Em, i).conj().T) for Em in Eks])
                Rk.append(w * op)

            Rk = sum(Rk)
            max_eig = np.linalg.eigvalsh(np.dot(Rk.conj().T, Rk))[-1]
            if max_eig > 0:
                Rks.append(Rk / np.sqrt(max_eig))
            else:
                Rks.append(Rk)
        return Rks

    @staticmethod
    def dutta_optimized(error_kraus: list, error_full: list, codes: list) -> list:
        x0 = []
        for Eks in error_kraus:
            for i in codes:
                chis = []
                for En in Eks:
                    chis.append(sum([MD([i.conj().T, Em.conj().T, En, i]) for Em in Eks]))

                p_weights = [Eks[j].P for j in range(len(chis))]
                if sum(p_weights) > 0:
                    X_av = np.average(chis, weights=p_weights)
                else:
                    X_av = np.average(chis)

                x0.append(1.0 / np.real(X_av) if X_av != 0 else 1.0)

        x0 = np.array(x0)

        def objective(w):
            R_kraus = Resurrection.parameterized(error_kraus, codes, w)
            fid, _ = Fidelity.entanglement(R_kraus, error_full, codes)
            return -fid

        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-3}
        )

        return Resurrection.parameterized(error_kraus, codes, result.x)