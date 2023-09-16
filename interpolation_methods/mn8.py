import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# data from file 8
x_points = np.array([ -1.2000000e+001,
     -1.0000000e+001,
     -8.0000000e+000,
     -6.0000000e+000,
     -4.0000000e+000,
     -2.0000000e+000,
      0.0000000e+000,
      2.0000000e+000,
      4.0000000e+000,
      6.0000000e+000,
      8.0000000e+000])

y_points = np.array([ -4.7169811e-002,
 -2.8571429e-002,
 -2.3809524e-002,
 -4.5454545e-002,
-1.0000000e-001,
 -1.6666667e-001,
-1.0000000e-001,
 -4.5454545e-002,
 -2.3809524e-002,
-1.4285714e-002,
 -9.4339623e-003])


def divided_diff(x, y):
    n = len(y)
    coef = np.zeros([n, n])

    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = \
                (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

    return coef

def newton_poly(coef, x_data, x):
    n = len(x_data) - 1
    p = coef[n]
    for k in range(1,n+1):
        p = coef[n-k] + (x -x_data[n-k])*p
    return p


a_s = divided_diff(x_points, y_points)[0, :]

x_new = np.arange(-12, 8.1, .1)
y_new = newton_poly(a_s, x_points, x_new)

def cubic_interp1d(x0, x, y):

    x = np.asfarray(x)
    y = np.asfarray(y)

    if np.any(np.diff(x) < 0):
        indexes = np.argsort(x)
        x = x[indexes]
        y = y[indexes]

    size = len(x)

    xdiff = np.diff(x)
    ydiff = np.diff(y)

    Li = np.empty(size)
    Li_1 = np.empty(size-1)
    z = np.empty(size)

    Li[0] = sqrt(2*xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0
    z[0] = B0 / Li[0]

    for i in range(1, size-1, 1):
        Li_1[i] = xdiff[i-1] / Li[i-1]
        Li[i] = sqrt(2*(xdiff[i-1]+xdiff[i]) - Li_1[i-1] * Li_1[i-1])
        Bi = 6*(ydiff[i]/xdiff[i] - ydiff[i-1]/xdiff[i-1])
        z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    i = size - 1
    Li_1[i-1] = xdiff[-1] / Li[i-1]
    Li[i] = sqrt(2*xdiff[-1] - Li_1[i-1] * Li_1[i-1])
    Bi = 0.0
    z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]


    i = size-1
    z[i] = z[i] / Li[i]
    for i in range(size-2, -1, -1):
        z[i] = (z[i] - Li_1[i-1]*z[i+1])/Li[i]

    index = x.searchsorted(x0)
    np.clip(index, 1, size-1, index)

    xi1, xi0 = x[index], x[index-1]
    yi1, yi0 = y[index], y[index-1]
    zi1, zi0 = z[index], z[index-1]
    hi1 = xi1 - xi0

    f0 = zi0/(6*hi1)*(xi1-x0)**3 + \
         zi1/(6*hi1)*(x0-xi0)**3 + \
         (yi1/hi1 - zi1*hi1/6)*(x0-xi0) + \
         (yi0/hi1 - zi0*hi1/6)*(xi1-x0)
    return f0

plt.figure(figsize = (12, 8))
plt.scatter(x_points, y_points)
plt.plot(x_points, y_points, 'bo')
plt.plot(x_new, y_new)
X_new = np.linspace(-12, 2.1, 201)
plt.plot(x_new, cubic_interp1d(x_new, x_points, y_points))
plt.show()
