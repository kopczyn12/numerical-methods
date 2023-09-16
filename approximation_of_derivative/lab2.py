import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_dt():
    """Calculating derivative from formula"""
    x = 0.5
    h = np.arange(20.)
    div = 5**h
    steps = 0.4/div

    dt = (np.log(x + steps) - np.log(x - steps))/(2*steps)
    return dt, h, steps

def abs_e(dt, real_dt):
    """Calcualte the abs error"""
    abs_error = np.abs((dt - real_dt))
    return abs_error

def create_data(h, steps, dt, abs_error):
    """Creating data - dataframe"""
    data = np.array([h, steps, dt, abs_error]).T
    df = pd.DataFrame(data, columns=['Steps', 'H', 'Dt', 'ABS_error'])
    df = df.set_index(['Steps'])
    df.to_csv('results.csv')
    print(df)

def plot_results(steps, abs_error):
    """Plotting results"""
    plt.figure()
    plt.plot(steps, abs_error,'-')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def find_k(abs_error):
    """Finding the k where error is the smallest"""
    k = np.min(abs_error)
    indexx = np.where(abs_error==k)[0][0]
    print ("Index of k where the error is the smallest: " , indexx)

def main():
    """Main function"""
    dt, h, steps = calculate_dt()
    real_dt = 1/0.5
    abs_error = abs_e(dt, real_dt)
    create_data(h, steps, dt, abs_error)
    find_k(abs_error)
    plot_results(steps, abs_error)

if __name__ == "__main__":
    main()