import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sympy import *
from typing import Tuple

T_a: float = 100.
h_: float = 0.05
h_prime: float = 0.05
dx: float = 0.5
dy: float = 0.5
grid_size: int = 21


def get_grid_temperature(grid: np.ndarray, x_pos: int, y_pos: int) -> float:
    return grid[x_pos, y_pos]


def create_grid(size: int) -> np.ndarray:
    grid: np.ndarray = np.ones((size, size)) * T_a
    grid[0, :] = np.linspace(400, 300, size)
    grid[-1, :] = np.linspace(300, 200, size)
    grid[:, 0] = np.linspace(400, 300, size)
    grid[:, -1] = np.linspace(300, 200, size)
    return grid


def create_system_of_linear_equations(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    size = grid.shape[0]
    n = (size - 2) ** 2
    A = np.zeros((n, n))
    b = np.zeros(n)

    for j in range(1, size - 1):
        for i in range(1, size - 1):
            # Get the index of this grid point in the x vector
            index = (j - 1) * (size - 2) + (i - 1)

            # Set b
            b[index] = h_prime * T_a * dx * dy

            # Set A
            A[index, index] = -4 - h_prime * dx * dy
            if i > 1:  # Connection to left node
                A[index, index - 1] = 1
            if i < size - 2:  # Connection to right node
                A[index, index + 1] = 1
            if j > 1:  # Connection to below node
                A[index, index - (size - 2)] = 1
            if j < size - 2:  # Connection to above node
                A[index, index + (size - 2)] = 1

            # Adjust b for boundary conditions
            if i == 1:
                b[index] -= grid[j, 0]
            if i == size - 2:
                b[index] -= grid[j, -1]
            if j == 1:
                b[index] -= grid[0, i]
            if j == size - 2:
                b[index] -= grid[-1, i]

    return A, b


if __name__ == "__main__":
    grid = create_grid(grid_size)

    A, b = create_system_of_linear_equations(grid)

    x = np.linalg.solve(A, b)

    grid[1:-1, 1:-1] = x.reshape(grid_size - 2, grid_size - 2)

    print(min(grid.reshape(-1)))
    print(np.argmax(min(grid.reshape(-1))))
    print(A)
    print(b)
    print(x)

    plt.imshow(grid, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar(label='Temperature')
    plt.show()