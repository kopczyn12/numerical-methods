import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#pkt poczatkowy, wartosc oryginalna
x_point = 0.5
y_point = np.log(x_point)
x_fun = np.arange(0,1,0.01)
y_fun = np.log(x_fun)


results = np.zeros((11,4))

def rozwiniecie(n,x_in):
    series = list(map(lambda i: ((x_in-1)**(2*i+1))/((2*i + 1) * ((x_in+1)**(2*i+1))), range(n+1)))
    result = sum(series)
    return 2*result

#iteracja dla 10 - wynik zbiega
for n in range(11):
    y = rozwiniecie(n,x_point)
    results[n][0] = n
    results[n][1] = y
    results[n][2] = y_point - y
    results[n][3] = 100*(np.abs((y_point-y)/y_point))


#tabela 
cols = ['Iteration', 'Results', 'Error', 'ABS Error']

df = pd.DataFrame(results, columns=cols)
print(df)


#wykresy
y_fun_n0 = rozwiniecie(0,x_fun)
y_fun_n3 = rozwiniecie(3,x_fun)
y_fun_n10 = rozwiniecie(10,x_fun)
 
fig, ax = plt.subplots()
fun_org = ax.plot(x_fun,y_fun,label ="org")
fun_n0 = ax.plot(x_fun,y_fun_n0,label="n = 0")
fun_n3 = ax.plot(x_fun,y_fun_n3,label="n = 3")
fun_n10 = ax.plot(x_fun,y_fun_n10, label="n =10 ")

ax.legend()
plt.show()