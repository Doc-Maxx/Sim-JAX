#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 13:44:14 2025

@author: Doc-Maxx
"""

import jax
import jax.numpy as jnp
from jax import jit
from matplotlib import pyplot as plt, cm
from tqdm import tqdm

N_ITERATIONS = 6000

NX = 41
NY = 41
DT = 0.0001

LENGTH_X = 1
LENGTH_Y = 1

c = 1

VISUALIZE = True
PLOT_EVERY_N_STEPS = 100

def update_Hx(Hx, Ez, dt, dy):
    Hx = Hx.at[1:-1,1:-1].set(
        Hx[1:-1,1:-1] - dt / dy * (Ez[:-2, 1:-1] - Ez[2:, 1:-1])
        )  
    return Hx

def update_Hy(Hy, Ez, dt, dx):
    Hy = Hy.at[1:-1,1:-1].set(
        Hy[1:-1,1:-1] - dt / dx * (Ez[1:-1,:-2] - Ez[1:-1,2:])
        )  
    return Hy

def update_Ez(Ez, Hx, Hy, Eps_i, Eps_r, omega, dt, dx, dy):
    Ez = Ez.at[1:-1,1:-1].set(
        Ez[1:-1,1:-1]*jnp.exp(- Eps_i[1:-1,1:-1] * omega * dt / Eps_r[1:-1,1:-1])
        + c**2 * dt / Eps_r[1:-1,1:-1] * ( 1 / dx * (Hy[1:-1,1:-1] - Hy[1:-1,:-2])
        +  1 / dy * (-Hx[1:-1,1:-1] + Hx[2:, 1:-1]) )
        )
    return Ez

def main():
    
    x = jnp.linspace(0, LENGTH_X, NX)
    y = jnp.linspace(0, LENGTH_Y, NY)
    
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    dx = LENGTH_X / (NX - 1)
    dy = LENGTH_Y / (NY - 1)
    
    Hx = jnp.zeros((NX,NY))
    Hy = jnp.zeros((NX,NY))
    Ez = jnp.zeros((NX,NY))
    
    Eps_i = jnp.ones((NX,NY)) * 0  # Decay
    Eps_r = jnp.ones((NX,NY)) * 1
    
    omega = 0.1 # no decay, so we don't need to set this to anything right now
    
    n_iter = 0 # Itneration number and manipulate the source
    
    source = jnp.sin(2 * jnp.pi * DT * n_iter * omega)
    Ez = Ez.at[-2,NX//3:-NX//3+1].set(source)
    
    fig = plt.figure(figsize=(11,7), dpi=100)
    
    for iteration_index in tqdm(range(N_ITERATIONS)):
        
        #Update H field
        
        Hx =  update_Hx(Hx, Ez, DT, dy)
        Hy = update_Hy(Hy, Ez, DT, dx)
        #print(Hx)
        #Enforce Boundary
        
        Hx = Hx.at[:,0].set(0)
        Hx = Hx.at[:,-1].set(0)
        Hx = Hx.at[0,:].set(0)
        Hx = Hx.at[-1,:].set(0)
        
        Hy = Hy.at[:,0].set(0)
        Hy = Hy.at[:,-1].set(0)
        Hy = Hy.at[0,:].set(0)
        Hy = Hy.at[-1,:].set(0)
        
        #Update E field
        
        Ez = update_Ez(Ez, Hx, Hy, Eps_i, Eps_r, omega, DT, dx, dy)
        
        #Enforce Boundary and Source
        source = jnp.sin(2 * jnp.pi * DT * n_iter * omega ) 
        Ez = Ez.at[-2,NX//3:-NX//3].set(source)
        
        Ez = Ez.at[:,0].set(0)
        Ez = Ez.at[:,-1].set(0)
        Ez = Ez.at[0,:].set(0)
        Ez = Ez.at[-1,:].set(0)
        
        #print(Ez)
        n_iter += 1
        
        
        if iteration_index % PLOT_EVERY_N_STEPS ==0 and VISUALIZE:
            # Contourf plot for pressure field with colorbar
            cf = plt.contourf(X, Y, Ez, alpha=0.5, cmap='turbo', levels=20)
            #plt.colorbar(cf, label='Z-Comp of Electric Field ')
            
            plt.show()
        
    
   


if __name__=="__main__":
    main()