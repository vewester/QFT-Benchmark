# %%
import pennylane as qml
from pennylane import numpy as np
from timeit import timeit
from memory_profiler import memory_usage


def qft(wires):
    num_qubits = len(wires)

    for i in range(num_qubits):
        qml.Hadamard(wires=i)
        for j in range(i + 1, num_qubits):
            angle = np.pi / (2 ** (j - i))
            qml.ControlledPhaseShift(phi=angle, wires=[wires[j], wires[i]])


def qft_wrapper(num_qubits):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(np.random.randint(2, size=num_qubits), wires=range(num_qubits))
        qft(wires=range(num_qubits))
        return qml.state()

    return circuit()


def benchmark_qft(max_qubits, repetitions=5):
    qubit_counts = []
    execution_times = []
    memory_usages = []

    for num_qubits in range(2, max_qubits + 1):
        exec_time = (
            timeit(lambda: qft_wrapper(num_qubits), number=repetitions) / repetitions
        )

        mem_usage = max(
            memory_usage(
                (qft_wrapper, (num_qubits,)), interval=0.1, include_children=True
            )
        )

        qubit_counts.append(num_qubits)
        execution_times.append(exec_time)
        memory_usages.append(mem_usage)

        print(
            f"Qubits: {num_qubits:>2}, Time: {exec_time:.6f} sec, Memory: {mem_usage:.2f} MB"
        )


def main():
    num_qubits = int(input("Benchmark QFT for qubits in range [2, n]\nn = "))
    benchmark_qft(num_qubits)


if __name__ == "__main__":
    main()
