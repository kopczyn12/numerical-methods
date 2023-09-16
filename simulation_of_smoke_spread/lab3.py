import numpy as np
from scipy.linalg import lu
import pandas as pd

#CONSTANTS 
Qa = 200
Ca = 2
Ws = 1500
E12 = 25
Qb = 300
Cb = 2
E23 = 50
E35 = 25
E34 = 50
Qc = 150
Qd = 350
Wg = 2500

A = np.array([[Qa+E12, -E12, 0, 0, 0],
              [-E12-Qa, 575, -E34, 0, 0],
              [0, -Qa-Qb-E23, E23+E35+E34+Qa+Qb, -E34, -E35],
              [0, 0, -E34-Qc, Qc+E34, 0],
              [0, 0, -Qd-E35, 0, Qd+E35]])

B = np.array([[Ws + Qa*Ca],
              [Cb*Qb],
              [0],
              [0],
              [Wg]])
#1
A_inv = np.linalg.inv(A)
C_1 = A_inv@B
df_c1 = pd.DataFrame(C_1)

#2
_, L, U = lu(A)
L_inv = np.linalg.inv(L)
U_inv = np.linalg.inv(U)
C_2 = U_inv@(L_inv@B)
df_c2 = pd.DataFrame(C_2)

#3

B_new = np.array([[800 + Qa*Ca],
                  [Cb*Qb],
                  [0],
                  [0],
                  [1200]])
C_2_new = U_inv@(L_inv@B_new)
df_c2_new = pd.DataFrame(C_2_new)

#4

A_inv_2 = np.dot(U_inv, L_inv)

#5
grill = 100 * (A_inv[3][4] * Wg) / C_1[3]
smokers = 100 * (A_inv[3][0] * Ws) / C_1[3]
street = 100 * (A_inv[3][0] * Qa * Ca + A_inv[3][1] * Qb * Cb) / C_1[3]

print('Exercise 1:\n')
print(df_c1)
print()
print(A)
print()
print('Exercise 2:\n')
print('Matrix L_inv')
print(L_inv)
print('Matrix U_inv')
print(U_inv)
print('CO2')
print(df_c2)
print()
print('Exercise 3:\n')
print(df_c2_new)
print()
print('Exercise 4:\n')
print(A_inv_2)
print()
print('Exercise 5:\n')
print(f"grill %: {grill}")
print(f"smokers %: {smokers}")
print(f"street %: {street}")