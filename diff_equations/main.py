import math
import numpy as np
from matplotlib import pyplot as plt

N = 25
t0 = 0
y0 = 2

def analitycznie(tk):
    h = (tk / N)
    y_out =[y0]
    t_i = [t0]

    for i in range(N):
        t_i.append(t_i[-1] + h)
        y_out.append(2*math.exp(t_i[-1]*(0.5 - t_i[-1])))

    return t_i, y_out

def f(t, y):
    return -2*y*t + 0.5*y

def Euler(tk):
    h = (tk / N)
    y_out = [y0]
    ti = [t0]

    for i in range(N):
        y_out.append(y_out[-1] + f(ti[-1], y_out[-1]) * h)
        ti.append(ti[-1] + h)
    return ti, y_out


def Heun(tk):
    h = (tk / N)
    y_out = [y0]
    ti = [t0]

    for i in range(N):
        y_0 = y_out[-1] + f(ti[-1], y_out[-1])*h
        y_out.append(y_out[-1] + ((f(ti[-1], y_out[-1]) + f(ti[-1] + h, y_0))/2) * h)
        ti.append(ti[-1] + h)
    return ti, y_out

def punkt_środka(tk):
    h = (tk / N)
    y_out = []
    y_out.append(y0)
    ti = [t0]

    for i in range(N):
        y_05 = y_out[-1] + f(ti[-1], y_out[-1]) * h/2
        y_out.append(y_out[-1] + f(ti[-1]+(h/2), y_05) * h)
        ti.append(ti[-1] + h)
    return ti, y_out



t_an, y_an = analitycznie(1.5)
T, Y = Euler(1.5)
t_heun, y_heun = Heun(1.5)
t_sr, y_sr = punkt_środka(1.5)

plt.scatter(t_an, y_an, s=8)
plt.plot(T, Y)
plt.plot(t_heun, y_heun)
plt.plot(t_sr, y_sr)
plt.legend(["analitycznie", "Euler", "Heun", "punkt srodka"])
plt.show()