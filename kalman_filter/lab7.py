import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg


def kalman_filter():
    """Implementation of the Kalman filter"""
    size = len(response[0])
    snn = np.array([[response[0, 0]], [response[1, 0]], [0], [0]])
    Pnn = np.identity(4) * 5
    gqg = G @ Q @ np.transpose(G)
    x_array = []
    y_array = []
    for n in range(size):
        sn1n = F @ snn
        Pn1n = F @ Pnn @ np.transpose(F) + gqg
        zn1n = H @ sn1n
        en1 = np.array([[response[0, n]], [response[1, n]]]) - zn1n
        Sn1 = H @ Pn1n @ np.transpose(H) + R
        Kn1 = Pn1n @ np.transpose(H) @ linalg.inv(Sn1)
        sn1n1 = sn1n + Kn1 @ en1

        Pn1n1 = (np.identity((Kn1 @ H).shape[0]) - Kn1 @ H) @ Pn1n
        x_array.append(sn1n1[0])
        y_array.append(sn1n1[1])
        snn = sn1n1
        Pnn = Pn1n1

    xs_array = [x_array[-1]]
    ys_array = [y_array[-1]]
    for n in range(0, 5):
        sn1n = F @ snn
        Pn1n = F @ Pnn @ np.transpose(F) + gqg
        zn1n = H @ sn1n
        en1 = np.array([snn[0], snn[1]]) - zn1n
        Sn1 = H @ Pn1n @ np.transpose(H) + R
        Kn1 = Pn1n @ np.transpose(H) @ linalg.inv(Sn1)
        sn1n1 = sn1n + Kn1 @ en1

        Pn1n1 = (np.identity((Kn1 @ H).shape[0]) - Kn1 @ H) @ Pn1n
        xs_array.append(sn1n1[0])
        ys_array.append(sn1n1[1])
        snn = sn1n1
        Pnn = Pn1n1
    return x_array, y_array, xs_array, ys_array


#loading data
dt = 0.01
n_max = int(12.5 / dt) + 1
response = []
with open('lab_7_dane/measurements8.txt', 'r') as f:
    response = [[float(num) for num in line.split()] for line in f]
response = np.transpose(response)

T = 1
Q = np.identity(2) * 0.25
R = np.identity(2) * 2.0
F = np.identity(4)
F[0, 2], F[1, 3] = T, T
G = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=float)
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

plt.plot(response[0, 0], response[1, 0], 'ro', label='Beginning of the trajectory')
plt.plot(response[0, -1], response[1, -1], 'bo', label='End of the trajectory')
plt.plot(response[0], response[1], 'gx', label='Trajectory loaded from file')
kfrx, kfry, kfrxs, kfrys = kalman_filter()
plt.plot(kfrx, kfry, 'r', label='Trajectory found by the Kalman Filter')
plt.plot(kfrxs, kfrys, 'o--', label='Estimated trajectory')
plt.plot(kfrxs[-1], kfrys[-1], '.', label='Plane after 5 sec')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.show()