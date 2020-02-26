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

def plot_microchip_data(df):
    accept_df = df[df['Accept or Reject']==1]
    reject_df = df[df['Accept or Reject']==0]
    plt.scatter(accept_df['Microchip Test 1'], accept_df['Microchip Test 2'], marker='+', color='black')
    plt.scatter(reject_df['Microchip Test 1'], reject_df['Microchip Test 2'], marker='o', color='yellow', edgecolors='black')
    plt.ylabel('Microchip Test 2 Score')
    plt.xlabel('Microchip Test 1 Score')
    plt.show()

def plot_best_fit(df, theta):
    pass_df = df[df['Pass or Fail']==1]
    fail_df = df[df['Pass or Fail']==0]

    theta_0 = theta[0].item()
    theta_1 = theta[1].item()
    theta_2 = theta[2].item()
    
    x_1 = df['Exam Score 1'].min()
    y_1 = (-1 / theta_2) * ((x_1*theta_1) + theta_0)
    x_2 = df['Exam Score 1'].max()
    y_2 = (-1 / theta_2) * ((x_2*theta_1) + theta_0)

    plt.scatter(pass_df['Exam Score 1'], pass_df['Exam Score 2'], marker='+', color='black')
    plt.scatter(fail_df['Exam Score 1'], fail_df['Exam Score 2'], marker='o', color='yellow', edgecolors='black')
    plt.plot([x_1, x_2], [y_1, y_2], color='blue') 
    plt.ylabel('Exam Score 2')
    plt.xlabel('Exam Score 1')

    plt.show()

def map_features(X, degree):
    '''Calculates polynomial features given a dataset
    
    Args:
        X (np.matrix): (m,n) training dataset
        degree (n): polynomial degree.
    
    Returns:
        X_p (np.matrix): training dataset with additional polynomial features
    '''
    X_p = np.matrix(np.ones(len(X))).T
    for i in range(1, degree+1):
        for j in range(0, i+1):
            x = np.multiply(np.power(X[:,0], i-j),np.power(X[:,1], j))
            X_p = np.append(X_p, x, axis=1)
    return X_p

def plot_contour(theta, df):
    accept_df = df[df['Accept or Reject']==1]
    reject_df = df[df['Accept or Reject']==0]

    plt.scatter(accept_df['Microchip Test 1'], accept_df['Microchip Test 2'], marker='+', color='black')
    plt.scatter(reject_df['Microchip Test 1'], reject_df['Microchip Test 2'], marker='o', color='yellow', edgecolors='black')
    plt.ylabel('Microchip Test 2 Score')
    plt.xlabel('Microchip Test 1 Score')
    
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros((len(u), len(v)))
    in_u = np.asmatrix(u).T
    in_v = np.asmatrix(v).T

    # Evaluate z = theta*x over the grid
    for i in range(0,len(u)):
        for j in range(0,len(v)):
            some_X = np.hstack((in_u[i,:], in_v[j,:]))
            X_mapped = map_features(some_X, 6)
            X_mapped = np.hstack((np.ones((len(X_mapped),1)), X_mapped))
            
            z[i,j] = np.dot(X_mapped, theta)

    # Transpose Z
    z = z.T
    
    plt.contour(u, v, z, 0)
    plt.show()
