'''Functions to plot data in for Linear Regression notebook'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_cost_function(J_vals):
    '''Plot cost function versus number of iterations'''
    plt.plot(np.arange(len(J_vals)), np.asarray(J_vals))
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    plt.show()

def plot_with_bestfit(x, y, theta):
    # turn matrices into 1d array
    y = np.squeeze(np.asarray(y))
    x = np.squeeze(np.asarray(x))
    coords = get_line_equation(theta, x)

    plt.scatter(x, y, marker='x', color='red')
    plt.plot(coords[0,:], coords[1,:], color='blue') 
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.title('Profit of a restaurant chain in different sized cities')

    plt.show()

def get_line_equation(theta, x):
    '''Plot the best fit line defined by theta
        Using linear equation in the form:
        y = mx + c
    '''
    c = theta.item(0)
    m = theta.item(1)

    x_0 = np.amin(x)
    y_0 = m*x_0 + c

    x_1 = np.amax(x)
    y_1 = m*x_1 + c
    
    coords = np.array([[x_0, x_1],
                      [y_0, y_1]])
    
    return coords
def plot_3d(x1, x2, y):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.zaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.scatter(x1, x2, y, c='red')
    ax.set_xlabel('Square footage')
    ax.set_ylabel('# of bedrooms')
    ax.set_zlabel('Price ($1,000s)')
    ax.view_init(20, -70)
    plt.title('House prices')
    plt.tight_layout()
    plt.show()

def plot_3d_with_bestfit(X, y, theta):
    '''
        Fit line equation using
        z = mx + ny + c
    '''
    c = theta[0,0]
    m = theta[1,0]
    n = theta[2,0]

    # Get x & y evenly sampled points
    X_plot = np.arange(np.amin(X[:,1]), np.amax(X[:,1]), 0.1)
    Y_plot = np.arange(np.amin(X[:,2]), np.amax(X[:,2]), 0.1)
    # Transform to a 2D mesh grid
    X_plot, Y_plot = np.meshgrid(X_plot, Y_plot)
    # Calculate Z values
    Z_plot = (X_plot*m + Y_plot*n + c) / 1000

    # Plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(np.squeeze(np.asarray(X[:,1])), np.squeeze(np.asarray(X[:,2])), y, c='red', marker='x')
    ax.plot_surface(X_plot, Y_plot, Z_plot)
    ax.view_init(20, -80)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_xlabel('Square footage')
    ax.set_ylabel('# of bedrooms')
    ax.set_zlabel('Price ($1,000s)')
    plt.title('Normalized linear regression')
    plt.show()