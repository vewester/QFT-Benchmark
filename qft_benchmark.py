# %%
import pennylane as qml
from pennylane import numpy as np
from timeit import timeit
from memory_profiler import memory_usage
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import markers


def qft(wires):
    """Manual implementation of the Quantum Fourier Transform (QFT)."""
    num_qubits = len(wires)
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
        for j in range(i + 1, num_qubits):
            angle = np.pi / (2 ** (j - i))
            qml.ControlledPhaseShift(phi=angle, wires=[wires[j], wires[i]])


def qft_wrapper(num_qubits):
    """Wraps the manual QFT in a PennyLane quantum node."""
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit():
        qft(wires=range(num_qubits))
        return qml.state()

    return circuit()


def pennylane_qft_wrapper(num_qubits):
    """Wraps PennyLane's native QFT in a quantum node."""
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit():
        qml.QFT(wires=range(num_qubits))
        return qml.state()

    return circuit()


def benchmark_qft(max_qubits, repetitions=5):
    """Benchmark both the manual and PennyLane QFT implementations."""
    data = []

    for num_qubits in range(3, max_qubits + 1):
        # Benchmark manual QFT
        exec_time_manual = (
            timeit(lambda: qft_wrapper(num_qubits), number=repetitions) / repetitions
        )
        # TODO: Figure out averaging memory usage
        mem_usage_manual = memory_usage(
            (qft_wrapper, (num_qubits,)),
            interval=0.1,
            include_children=True,
            max_usage=True,
        )

        # Benchmark PennyLane QFT
        exec_time_pennylane = (
            timeit(lambda: pennylane_qft_wrapper(num_qubits), number=repetitions)
            / repetitions
        )
        mem_usage_pennylane = memory_usage(
            (pennylane_qft_wrapper, (num_qubits,)),
            interval=0.1,
            include_children=True,
            max_usage=True,
        )

        # Store the results
        data.append(
            {
                "num_qubits": num_qubits,
                "exec_time_manual": exec_time_manual,
                "mem_usage_manual": mem_usage_manual,
                "exec_time_pennylane": exec_time_pennylane,
                "mem_usage_pennylane": mem_usage_pennylane,
            }
        )

        print(
            f"Qubits: {num_qubits:>2}, "
            f"Manual Time: {exec_time_manual:.6f} sec, Manual Memory: {mem_usage_manual:.2f} MB, "
            f"PennyLane Time: {exec_time_pennylane:.6f} sec, PennyLane Memory: {mem_usage_pennylane:.2f} MB"
        )

    # Convert the results to a pandas DataFrame for easy plotting and analysis
    df = pd.DataFrame(data)

    # Plot execution time comparison
    plt.plot(
        df["num_qubits"],
        df["exec_time_manual"],
        label="Manual QFT Time",
        color="blue",
        marker=markers.MarkerStyle(marker="o"),
    )
    plt.plot(
        df["num_qubits"],
        df["exec_time_pennylane"],
        label="PennyLane QFT Time",
        color="green",
        marker=markers.MarkerStyle(marker="s"),
    )
    plt.xlabel("Number of Qubits")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.show()

    # Plot memory usage comparison
    plt.plot(
        df["num_qubits"],
        df["mem_usage_manual"],
        label="Manual QFT Memory",
        color="blue",
        marker=markers.MarkerStyle(marker="o"),
    )
    plt.plot(
        df["num_qubits"],
        df["mem_usage_pennylane"],
        label="PennyLane QFT Memory",
        color="green",
        marker=markers.MarkerStyle(marker="s"),
    )
    plt.xlabel("Number of Qubits")
    plt.ylabel("Memory Usage (MB)")
    plt.legend()
    plt.show()


def main():
    num_qubits = int(input("Benchmark QFT for qubits in range [3, n]\nn = "))
    repetitions = int(input("Number of repetitions to average for each benchmark\nr = "))
    benchmark_qft(num_qubits, repetitions)


if __name__ == "__main__":
    main()

# %%
