
import numpy as np
from matplotlib import pyplot as plt


a2= -3.85
a1 = 2.7
a0 = -0.45

# Matrixes
A = np.mat([[-a2, -a1, -a0],
               [1, 0, 0],
               [0, 1, 0]])

B = np.mat([[1], [0], [0]])
C = np.mat(np.ones((1,3)))
D = np.mat([0])

x0 = np.mat(np.zeros(B.shape))



def plot_step_response(A, B, C, D, x, len_of_signal):
    y = np.array([])
    t = np.array(range(len_of_signal))


    for i in range(len_of_signal):
        y_n = np.array(C*x + D)
        x = A * x + B
        y = np.append(y, y_n.flatten(),axis=0)
    u = np.ones(t.shape)

    plt.stem(t,u,  markerfmt='C3x')
    plt.stem(t,y)
    plt.grid()
    plt.title("Odpowiedz skokowa")
    plt.rcParams['figure.figsize'] = [18, 8]
    plt.legend(["sygnal pobudzający u [n]", "sygnal wyjsciowy"], loc ="upper left", fontsize='xx-large')
    plt.xlabel('probka n')
    plt.ylabel('sygnal')
    plt.show()

def plot_step_response_2(A, B, C, D, x, len_of_signal, F):
    y = np.array([])
    t = np.array(range(len_of_signal))
    u = np.array([])

    for i in range(len_of_signal):
        y_n = np.array(C*x + D)
        
        u_k_n = 1 + F * x
        u = np.append(u,u_k_n)

        
        x = A * x + B
        y = np.append(y, y_n.flatten(),axis=0)

    
    plt.scatter(t,u,c="red")
    plt.stem(t,y)
    plt.grid()
    plt.title("Odpowiedz skokowa")
    
    plt.rcParams['figure.figsize'] = [18, 8]
    plt.legend(["sygnal sterujacy u [n]", "sygnal wyjsciowy"], loc ="lower right", fontsize='xx-large')
    plt.xlabel('probka n')
    plt.ylabel('sygnal')


    plt.show()

def printMatrix(title: str, matrix: np.ndarray):
    print(title)
    print('\n'.join([''.join(['{:11}'.format(round(item, 7)) for item in row]) for row in matrix]))
    print('\n')

plot_step_response(A,B,C,D,x0, 10)
# Układ nie jest stabilny


c1 = 10
c2 = 1
num_of_iterations = 100

Q = c1 * np.identity(A.shape[0]) 
R = np.array([c2]) 
P = np.mat(np.identity(A.shape[0]))
Q = np.mat(Q)
R = np.mat(R)


# Finding P
for i in range(num_of_iterations):
    P_new =  Q + A.T * (P - P*B * np.linalg.inv(R + B.T * P * B) * B.T*P) * A
    P = P_new 


printMatrix("Macierz P: ", np.array(P))
# Wyznaczona macierz P

# Finding F
F = np.linalg.inv(R + B.T * P * B) * B.T * P * A

A_new = A - B * F

F_new = np.linalg.inv(R + B.T * P * B) * B.T * P * A_new


plot_step_response_2(A_new,B, C,D, x0, 50, F_new)

# Gdy c2 zwiększamy układ wolniej dochodzi do stanu ustalonego
# Gdy mamy bardzo duże c1 układ szybko się ustala, ważny jest stosunek c1/c2 im większy on jest tym układ ustala się szybko