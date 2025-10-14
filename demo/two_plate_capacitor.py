#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 07:04:18 2025

@author: Doc-Maxx
"""

'''
This code is inspired by some of Zhengtao Gan's online CFD course.

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
        V = V.at[1:-1,1:-1].set(((dy**2 * (Vn[2:,1:-1] + Vn[:-2,1:-1]) +
                        dx**2 * (Vn[1:-1,2:] + Vn[1:-1,:-2])) /
                        (2 * (dx**2 + dy**2))))
        
        #jax.debug.print("{V}",V=V)
        V= jax.lax.dynamic_update_slice(V,plate1, (BX, BY))
        V= jax.lax.dynamic_update_slice(V,plate2, (BX,NY- BY))
        #jax.debug.print("{V}",V=V)
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
        
        #Compute Condition
        norm = (jnp.sum(jnp.abs(V[:])-jnp.abs(Vn[:])) / jnp.sum(jnp.abs(Vn[:])))
        return norm > l1norm_target
    
    def body(loop_state):
        #Unpack Loop State
        V = loop_state[0]
        dx = loop_state[2]
        dy = loop_state[3]
        BX = loop_state[4]
        BY = loop_state[5]
        
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
dx = 2 / (NX - 1)
dy = 2 / (NY - 1)

V = jnp.zeros((NX,NY))

x = jnp.linspace(-10,10,NX)
y = jnp.linspace(-10,10,NY)
V = V.at[0,:].set(0)  
V = V.at[-1,:].set(0)
V = V.at[:,0].set(0)  
V = V.at[:,-1].set(0)
V = V.at[:,1].set(V[:,0])  
V = V.at[:,-2].set(V[:,-1])
V = V.at[1,:].set(V[0,:])  
V = V.at[-2,:].set(V[-1,:])

BX = jnp.int32(NX/3)
BY = jnp.int32(NY/3)

plate1 = jnp.ones((NX-2*BX,1))
plate2 = jnp.ones((NX-2*BX,1))*(-1)

V= jax.lax.dynamic_update_slice(V,plate1, (BX, BY))
V= jax.lax.dynamic_update_slice(V,plate2, (BX, -BY-1))

result = laplace2d_jax2_cap_plate(V, dx, dy, BX, BY, 1e-9) 

plot2D(x, y,result[0].T )
