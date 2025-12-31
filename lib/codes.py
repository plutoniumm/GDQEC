from itertools import combinations as nCr
import numpy as np


# 0L = |001> + |010> + |100>
# 1L = |111>
def _dutta3():
    dutta_3_0 = np.zeros(8)
    dutta_3_1 = np.zeros(8)

    dutta_3_0[1] = 1
    dutta_3_0[2] = 1
    dutta_3_0[4] = 1
    dutta_3_1[7] = 1

    return np.array(
        [dutta_3_0 / np.linalg.norm(dutta_3_0), dutta_3_1 / np.linalg.norm(dutta_3_1)]
    )


# LEUNG STD [4, 1] code
# |0L> = |0000> + |1111>
# |1L> = |0011> + |1100>
def _leung():
    leung_0 = np.zeros(16)
    leung_1 = np.zeros(16)

    leung_0[0] = 1
    leung_0[-1] = 1
    leung_1[3] = 1
    leung_1[12] = 1

    return np.array(
        [leung_0 / np.linalg.norm(leung_0), leung_1 / np.linalg.norm(leung_1)]
    )


leung = _leung()


# PERMUTATION INVARIANT [5, 1] code
# |0L> = |00001> + |00010> + |00100> + |01000> + |10000>
# |1L> = XXXXX|0L>
def _dutta5():
    dutta_5_0 = np.zeros(32)
    dutta_5_1 = np.zeros(32)
    for i in range(0, 5):
        dutta_5_0[1 << i] = 1
        dutta_5_1[31 - (1 << i)] = 1

    return np.array(
        [dutta_5_0 / np.linalg.norm(dutta_5_0), dutta_5_1 / np.linalg.norm(dutta_5_1)]
    )


dutta_5 = _dutta5()


# [[5, 1, 3]] stabilizer code
# 0L = {\frac {1}{4}} 0b00000 + 0b10010 + 0b01001 + 0b10100 + 0b01010 - 0b11011 - 0b00110 - 0b11000 - 0b11101 - 0b00011 - 0b11110 - 0b01111 - 0b10001 - 0b01100 - 0b10111 + 0b00101
# [0, 18, 9, 20, 10, -27, -6, -24, -29, -3, -30, -15, -17, -12, -23, 5]
# 1L is just XXXXX|0L>
def _513():
    _0L = np.zeros(32)
    _1L = np.zeros(32)

    _0L_keys = [0, 18, 9, 20, 10, -27, -6, -24, -29, -3, -30, -15, -17, -12, -23, 5]
    _1L_keys = [31, 13, 22, 11, 21, -4, -25, -7, -2, -28, -1, -16, -14, -19, -8, 26]

    for i in range(len(_0L_keys)):
        _0key = _0L_keys[i]
        _1key = _1L_keys[i]

        _0L[np.abs(_0key)] = np.sign(_0key)
        _1L[np.abs(_1key)] = np.sign(_1key)

    _0L[0] = 1
    return np.array([_0L, _1L]) / 4


def _rep3_X():
    K0 = np.array([1, 0])
    K1 = np.array([0, 1])

    rep3_X_0 = np.kron(np.kron(K0, K0), K0)
    rep3_X_1 = np.kron(np.kron(K1, K1), K1)

    return np.array([rep3_X_0, rep3_X_1])


def _rep3_Z():
    K0 = np.array([1, 0])
    K1 = np.array([0, 1])

    Kp = (K0 + K1) / np.sqrt(2)
    Km = (K0 - K1) / np.sqrt(2)

    rep3_Z_0 = np.kron(np.kron(Kp, Kp), Kp)
    rep3_Z_1 = np.kron(np.kron(Km, Km), Km)

    return np.array([rep3_Z_0, rep3_Z_1])


nudge = 0.05


def Code(key):
    if key == "D3":
        return _dutta3()
    elif key == "leung":
        return _leung()

    elif key == "D5":
        return _dutta5()
    elif key == "513":
        return _513()

    elif key == "XXX":
        return _rep3_X()
    elif key == "XX2":
        code = _rep3_X() + nudge
        code = code / np.linalg.norm(code, axis=1, keepdims=True)
        print(code)
        return code

    elif key == "ZZZ":
        return _rep3_Z()
    elif key == "ZZ2":
        code = _rep3_Z() + nudge
        return code / np.linalg.norm(code, axis=1, keepdims=True)

    else:
        raise ValueError(f"Unknown code: {key}")
