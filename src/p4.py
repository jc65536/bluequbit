from collections.abc import Iterable
from typing import cast

from qiskit import QuantumCircuit

from quimb.tensor import Tensor, TensorNetwork, CircuitPermMPS, Circuit
from quimb.tensor.tensor_1d import TensorNetwork1DVector
import numpy as np

from classical_sim.hamming_weight_2D import GateList, IGate, Unitary, compute_lightcone
from util import load_qasm


P4_PATH = "circuits/P4_golden_mountain.qasm"


def tensor_network():
    qc = Circuit.from_openqasm2_file(P4_PATH)

    print(f"N: {qc.N}")

    for b in qc.sample(10):
        print(b)


def load_gatelist() -> tuple[int, GateList[Unitary]]:
    """
    Returns
    -------
    N, U: tuple[int, GateList[Unitary]]
        N is the number of qubits, U is a list of IGate[Unitary]. See
        classical_sim.hamming_weight_2D for the definition of these data
        structures.
    """
    qc = load_qasm(P4_PATH)

    N = qc.num_qubits

    U: GateList[Unitary] = []

    for ins in qc.data:
        idx: list[int] = [qc.find_bit(b)[0] for b in ins.qubits]
        mat: Unitary = ins.matrix
        name: str = ins.name
        params: list[float] = ins.params
        label = f"{name}({", ".join(map(str, params))})"

        U.append(IGate(mat, idx, label))

    return N, U


def lightcone():
    """
    Not used directly in the code, but allows you to compute the lightcone
    (it'll most likely be the set of all qubits)
    """
    N, U = load_gatelist()

    depth_map = np.zeros(N, dtype=np.int32)

    for g in U:
        if len(g.idx) == 2:
            i, j = g.idx
            depth_map[i] += 1
            depth_map[j] += 1

    print(f"Depth map: {depth_map}")

    print(f"len(U): {len(U)}")

    for part in range(len(U) // 71):
        print(f"Part {part}")
        part_start = part * 71
        part_end = (part + 1) * 71
        for b in compute_lightcone(U[part_start:part_end], N, 1):
            print(f"{b.R_begin}..{b.R_end}")


def sim_gatelist(
    N: int,
    U: GateList[Unitary],
    psi0: str | None = None,
    shots: int = 1000,
) -> dict[str, float]:
    """
    Parameters
    ----------
    N: int
        Number of qubits
    U: GateList[Unitary]
        The circuit to simulate
    psi0: str | None
        The initial state as a bitstring
    shots: int

    Returns
    -------
    counts: dict[str, float]
        For a bitstring b, counts[b] = the number of times b was sampled
    """
    qc = Circuit(N)

    if psi0 is not None:
        for i, x in enumerate(psi0):
            if x == "1":
                qc.x(i)

    for g in U:
        qc.apply_gate_raw(g.gate, g.idx)

    counts: dict[str, float] = {}

    for b in qc.sample(shots):
        counts.setdefault(b, 0)
        counts[b] += 1

    return counts


def combine_sim_counts(
    N: int,
    U: GateList[Unitary],
    states: dict[str, float],
    shots: int = 1000,
) -> dict[str, float]:
    """
    Parameters
    ----------
    states: dict[str, float]
        A dictionary of initial states (as bitstrings) and their probabilities.

    Returns
    -------
    all_counts: dict[str, float]
        Maps sampled bitstrings to weighted counts. For each initial state psi,
        its counts are weighted by states[psi].
    """
    all_counts: dict[str, float] = {}

    for p0, scale in states.items():
        print(f"Simulating {p0}")
        counts = sim_gatelist(N, U, p0, shots)
        for k, v in counts.items():
            all_counts.setdefault(k, 0)

            # Counts are weighted by the probability of the initial state
            all_counts[k] += v * scale

    return all_counts


def main():
    shots = 1000
    N, U = load_gatelist()
    print("GateList loaded")

    states = {"0" * N: 1.0}

    for i in range(len(U) // 71):
        print(f"Part {i}")
        counts = combine_sim_counts(
            N, U[71 * i:71 * (i + 1)], states, shots=shots)

        # Some stats for your viewing pleasure
        print(f"len(counts): {len(counts)}")
        a = np.array(list(counts.values()))
        print(f"avg(counts): {np.mean(a)}")
        print(f"stddev(counts): {np.std(a)}")

        # Uniform layers just scramble the states without enhancing peakedness
        if len(counts) >= 0.8 * len(states) * shots:
            print("Uniform")

        # Deterministic layers have only one possible output, enhancing peakedness
        if len(counts) == 1:
            print("Deterministic")

        most_likely_bitstrings = {k: v for k, v in sorted(
            counts.items(), key=lambda t: -t[1])[:10]}
        count_sum = sum(most_likely_bitstrings.values())
        states = {k: v / count_sum for k, v in most_likely_bitstrings.items()}

        print(states)
        print(flush=True)


def pp():
    states = {
        '000100101101001110000110011010111101111111010011': 0.15085570167537382,
        '000100101101001110000110011010111101111111010110': 0.11916171260433556,
        '011100101101001110000110011010111101111111010110': 0.10571068275986309,
        '001100101101001110000110011010111101111111010011': 0.10390920554854981,
        '010100101101001110000110011010111101111111010110': 0.1022518465141416,
        '000001001101001110000110011010111101111111010110': 0.08776796973518285,
        '000001001101001110000110011010111101111111010011': 0.08532997057587222,
        '010100101101001110000110011010111101111111010011': 0.08290398126463701,
        '011100101101001110000110011010111101111111010011': 0.08250765627814809,
        '001100101101001110000110011010111101111111010110': 0.07960127304389598
    }

    consensus = ""

    for i in range(48):
        a = [0.0, 0.0]
        for x, p in states.items():
            a[int(x[i])] += p
        print(a)
        if a[0] > a[1]:
            consensus += "0"
        else:
            consensus += "1"
    
    print(consensus)


if __name__ == "__main__":
    pp()
