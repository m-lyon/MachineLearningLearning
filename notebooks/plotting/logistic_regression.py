'''Functions to plot data in for Logistic Regression notebook'''

import numpy as np
import matplotlib.pyplot as plt

def plot_cost_function(J_vals):
    plt.plot(np.arange(len(J_vals)), J_vals)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    plt.show()

def plot_data(df):
    pass_df = df[df['Pass or Fail']==1]
    fail_df = df[df['Pass or Fail']==0]

    plt.scatter(pass_df['Exam Score 1'], pass_df['Exam Score 2'], marker='+', color='black')
    plt.scatter(fail_df['Exam Score 1'], fail_df['Exam Score 2'], marker='o', color='yellow', edgecolors='black')
    plt.ylabel('Exam 2 Score')
    plt.xlabel('Exam 1 Score')
    plt.show()