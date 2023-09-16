import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def function_1(x: float, y: float, z: float) -> float:
    return -10 * x + 10 * y


def function_2(x: float, y: float, z: float) -> float:
    return 28 * x - y - x * z


def function_3(x: float, y: float, z: float) -> float:
    return -8/3 * z + x * y

def RK4_3_equations(f1: Callable[[float, float, float], float],
                    f2: Callable[[float, float, float], float],
                    f3: Callable[[float, float, float], float],
                    t_0: int, t_k: int, h: float,
                    x_0: int, y_0: int, z_0: int) -> tuple[np.array, np.array, np.array]:
    num_of_steps: int = int(t_k/h)
    x: np.array = np.zeros(num_of_steps+1)
    y: np.array = np.zeros(num_of_steps+1)
    z: np.array = np.zeros(num_of_steps+1)
    x[0] = x_0
    y[0] = y_0
    z[0] = z_0

    t: np.array = np.arange(t_0, t_k, h)

    for i in range(num_of_steps):
        k1_x = h * f1(x[i], y[i], z[i])
        k1_y = h * f2(x[i], y[i], z[i])
        k1_z = h * f3(x[i], y[i], z[i])

        k2_x = h * f1(x[i] + k1_x / 2, y[i] + k1_y / 2, z[i] + k1_z / 2)
        k2_y = h * f2(x[i] + k1_x / 2, y[i] + k1_y / 2, z[i] + k1_z / 2)
        k2_z = h * f3(x[i] + k1_x / 2, y[i] + k1_y / 2, z[i] + k1_z / 2)

        k3_x = h * f1(x[i] + k2_x / 2, y[i] + k2_y / 2, z[i] + k2_z / 2)
        k3_y = h * f2(x[i] + k2_x / 2, y[i] + k2_y / 2, z[i] + k2_z / 2)
        k3_z = h * f3(x[i] + k2_x / 2, y[i] + k2_y / 2, z[i] + k2_z / 2)

        k4_x = h * f1(x[i] + k3_x, y[i] + k3_y, z[i] + k3_z)
        k4_y = h * f2(x[i] + k3_x, y[i] + k3_y, z[i] + k3_z)
        k4_z = h * f3(x[i] + k3_x, y[i] + k3_y, z[i] + k3_z)

        x[i + 1] = x[i] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        y[i + 1] = y[i] + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
        z[i + 1] = z[i] + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6

    return x, y, z


if __name__ == "__main__":
    initial_t: int = 0  
    final_t: int = 25  
    step: float = 0.03125
    initial_x, initial_y, initial_z = 5, 5, 5

    t: np.array = np.arange(initial_t, final_t, step)

    x_t, y_t, z_t = RK4_3_equations(function_1, function_2, function_3,
                    initial_t, final_t, step,
                    initial_x, initial_y, initial_z)

    plt.plot(t, x_t[:-1], color="r")
    plt.xlabel("Time values")
    plt.ylabel("X values")
    plt.show()

    plt.plot(t, y_t[:-1], color="g")
    plt.xlabel("Time values")
    plt.ylabel("Y values")
    plt.show()

    plt.plot(t, z_t[:-1], color="b")
    plt.xlabel("Time values")
    plt.ylabel("Z values")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_t[:-1], y_t[:-1], z_t[:-1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
