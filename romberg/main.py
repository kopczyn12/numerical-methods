import numpy as np

def trapezoid_intervals(fun, a, b, n):
    h = (b - a) / n
    integral = 0.5 * h * (fun(a) + fun(b))
    for i in range(1, n):
        integral += fun(a + i * h) * h
    return integral

def func(x):
    return (-0.06452) * x**4 + 0.5432 * x**3 - 0.7523 * x**2 - 3.132 * x + 2.756

def romberg_error(r_table, row):
    if row == 0:
        error = 100
    else:
        error = np.abs((r_table[row, row] - r_table[row, row - 1]) / r_table[row, row]) * 100.
    return error

def romberg(fun, a, b, steps):
    romberg_table = np.zeros((steps, steps), dtype=np.float64)
    
    for row in range(steps):
        romberg_table[row, 0] = trapezoid_intervals(fun, a, b, 2**row)
        
        for col in range(row):
            romberg_table[row, col + 1] = (4**(col + 1) * romberg_table[row, col] - romberg_table[row - 1, col]) / (4**(col + 1) - 1)

        if romberg_error(romberg_table, row) < 2:
            romberg_table = np.delete(romberg_table, [i for i in range(row + 1, steps)], 1)
            romberg_table = np.delete(romberg_table, [i for i in range(row + 1, steps)], 0)
            break

    return romberg_table

def func_gauss(a, b, x):
    result = (-0.06452) * ((b + a) / 2 + x * (b - a) / 2)**4 + 0.5432 * ((b + a) / 2 + x * (b - a) / 2)**3 - 0.7523 * ((b + a) / 2 + x * (b - a) / 2)**2 - 3.132 * ((b + a) / 2 + x * (b - a) / 2) + 2.756
    return result * (b - a) / 2

def gauss(a, b):
    c0, c1, c2 = 5 / 9, 8 / 9, 5 / 9
    x0, x1, x2 = -np.sqrt(3 / 5), 0, np.sqrt(3 / 5)
    return c0 * func_gauss(a, b, x0) + c1 * func_gauss(a, b, x1) + c2 * func_gauss(a, b, x2)

def main():
    num_of_steps = 10
    romberg_integrals = romberg(func, -3, 7, num_of_steps)
    romberg_solution = romberg_integrals[-1, -1]
    
    print(romberg_integrals)
    print(f'Romberg result: {romberg_solution}')
    
    gauss_integral = gauss(-3, 7)
    print(f'Gauss 3-point quadrature result: {gauss_integral}')

if __name__=="__main__":
    main()