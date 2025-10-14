#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 07:04:18 2025

@author: Doc-Maxx
"""

'''
This code reuses some code from Zhengtao Gan's online CFD course.

The physical setup is from N. Giordano & H. Nakanishi, Computational Physics 2nd Ed.
'''


import jax
import jax.numpy as jnp
from jax import jit

from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import Axes3D

def plot2D(x,y,p):  # define a function for visulizing 2d plot
    fig = plt.figure(figsize = (11,7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    # The '111' means a grid of 1 row and 1 column and this subplot is the first one.
    X, Y = jnp.meshgrid(x,y)
    surf = ax.plot_surface(X,Y,p,cmap=cm.viridis)
    ax.view_init(elev=25.0, azim=145)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$');

def laplace2d_jax2_cap_plate(V,dx,dy,BX, BY, l1norm_target):
    
    #pack the initial loop State
    init = [V, jnp.zeros((NX,NY)) ,dx, dy, BX, BY, l1norm_target]
    
    def stepper(V, dx, dy, BX, BY):
        #steps Laplace solver :)
        Vn = V.copy()
        
        # Finite difference method to compute the updated voltage within each cell
        V = V.at[1:-1,1:-1].set(((dy**2 * (Vn[2:,1:-1] + Vn[:-2,1:-1]) +
                        dx**2 * (Vn[1:-1,2:] + Vn[1:-1,:-2])) /
                        (2 * (dx**2 + dy**2))))
        
        # Enforce plate voltages 
        V= jax.lax.dynamic_update_slice(V,plate1, (BX, BY))
        V= jax.lax.dynamic_update_slice(V,plate2, (BX,NY- BY))
        
        # Enforce Boundary Conditions
        V = V.at[0,:].set(0)  
        V = V.at[-1,:].set(0)
        V = V.at[:,0].set(0)  
        V = V.at[:,-1].set(0)
        V = V.at[:,1].set(V[:,0])  
        V = V.at[:,-2].set(V[:,-1])
        V = V.at[1,:].set(V[0,:])  
        V = V.at[-2,:].set(V[-1,:])
        
        return V, Vn
    
    def condition(loop_state):
        #Unpack Loop State
        V = loop_state[0]
        Vn = loop_state[1]
        l1norm_target = loop_state[-1]
        
        ''' 
        Compute "norm" to compare the new voltage matrix to the previous one
        This solver uses a relaxational method. Each step makes finer adjustments
        to the voltage distribution. Once we reach an abritrarily small difference
        between each step, we send the signal to terminate the loop.
        '''
        norm = (jnp.sum(jnp.abs(V[:])-jnp.abs(Vn[:])) / jnp.sum(jnp.abs(Vn[:])))
        return norm > l1norm_target
    
    def body(loop_state):
        #Unpack Loop State
        V = loop_state[0]
        dx = loop_state[2]
        dy = loop_state[3]
        BX = loop_state[4]
        BY = loop_state[5]
        
        # Solve Step
        V, Vn = stepper(V, dx, dy, BX, BY)
        
        #Repack Loop State
        loop_state = [V, Vn, dx, dy, BX, BY, loop_state[-1]]
        return loop_state
        
    result = jax.lax.while_loop(condition,
                                body,
                                init)
    return result

# Set up simulation parameters
NX = 110
NY = 101

L1NORM_TARGET = 1e-6

dx = 2 / (NX - 1)
dy = 2 / (NY - 1)

x = jnp.linspace(-10,10,NX)
y = jnp.linspace(-10,10,NY)

# Set up initial conditions for the voltage
V = jnp.zeros((NX,NY))

# Set Boundary conditions
V = V.at[0,:].set(0)  
V = V.at[-1,:].set(0)
V = V.at[:,0].set(0)  
V = V.at[:,-1].set(0)
V = V.at[:,1].set(V[:,0])  
V = V.at[:,-2].set(V[:,-1])
V = V.at[1,:].set(V[0,:])  
V = V.at[-2,:].set(V[-1,:])

# BX and BY set the location of each plate
BX = jnp.int32(NX/3)
BY = jnp.int32(NY/3)

# Set plate voltages to 1 and -1.
# Each plate is a vector we will slice into the initial voltage matrix
plate1 = jnp.ones((NX-2*BX,1))
plate2 = jnp.ones((NX-2*BX,1))*(-1)

# Slice the plates into the voltage matrix
V= jax.lax.dynamic_update_slice(V,plate1, (BX, BY))
V= jax.lax.dynamic_update_slice(V,plate2, (BX, -BY-1))

# Compute the result and plot it.
result = laplace2d_jax2_cap_plate(V, dx, dy, BX, BY, L1NORM_TARGET) 

plot2D(x, y,result[0].T )
