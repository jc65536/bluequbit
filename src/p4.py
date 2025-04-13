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
        counts = combine_sim_counts(N, U[71 * i:71 * (i + 1)], states, shots=shots)
        print(f"len(counts): {len(counts)}")
        a = np.array(list(counts.values()))
        print(f"avg(counts): {np.mean(a)}")
        print(f"stddev(counts): {np.std(a)}")

        if len(counts) >= 0.8 * len(states) * shots:
            print("Uniform")
        if len(counts) == 1:
            print("Deterministic")

        most_likely_bitstrings = {k: v for k, v in sorted(counts.items(), key=lambda t: -t[1])[:10]}
        count_sum = sum(most_likely_bitstrings.values())
        states = {k: v / count_sum for k, v in most_likely_bitstrings.items()}
        print(states)
        print(flush=True)


if __name__ == "__main__":
    main()
