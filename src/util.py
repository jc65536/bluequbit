import qiskit.qasm2 as qasm

import bluequbit

import argparse

from qiskit import QuantumCircuit
from bluequbit import BQClient
from dotenv import dotenv_values

config = dotenv_values()


def bq_client() -> BQClient:
    return bluequbit.init(config["API_KEY"])


def load_qasm() -> QuantumCircuit:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    ns = parser.parse_args()
    return qasm.load(vars(ns)["input_file"])
