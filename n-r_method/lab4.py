import numpy as np
import matplotlib.pyplot as plt

# DATA
x = np.arange(-4, 1, 0.001)
y1 = -x*x - 5*x - 7
x1 = np.arange(-4, 0.199, 0.001)
y21 = (-3*x1*x1)/(1 - 5*x1)
x2 = np.arange(0.201, 1, 0.002)
y22 = (-3*x2*x2)/(1 - 5*x2)

# PLOT DATA
labels = ["y = -x^2 - 5x - 7", "y = -3x^2 + 5xy"]
plt.plot(x, y1)
plt.plot(x1, y21, color='red')
plt.plot(x2, y22, color='red')
plt.legend(labels)

#ITERATIONS
iterr = 10
xx = 0
for i in range(iterr):
    yy = -xx*xx - 5*xx - 7
    xx = (yy + 3*xx*xx)/(5*yy)
print(f"P3 solution after {iterr} iterations: {xx, yy}")

# N-R
starting_points = [(-4, -3), (-2, -1), (0, -7)]
for i in range(len(starting_points)):
    xx = starting_points[i][0]
    yy = starting_points[i][1]
    for j in range(iterr):
        df1_po_x = 2*xx + 5
        df1_po_y = 1
        df2_po_x = 6*xx - 5*yy
        df2_po_y = -5*xx + 1
        jakob = (df1_po_x*df2_po_y) - (df1_po_y*df2_po_x)
        f1 = yy + xx*xx + 5*xx + 7
        f2 = yy + 3*xx*xx - 5*xx*yy
        xx = xx - (f1*df2_po_y - f2*df1_po_y)/jakob
        yy = yy - (f2*df1_po_x - f1*df2_po_x)/jakob
    print(f"Solution by method N-R after {iterr} iterations are {starting_points[i], xx, yy}")
plt.show()
