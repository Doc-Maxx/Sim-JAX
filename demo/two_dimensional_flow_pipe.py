#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 05:44:35 2025

@author: Doc-Maxx
"""

"""
Here we will simulate a two dimensional flow.

The flow will be incompressible. 

Each velocity component will be solved using a finite difference method.
The pressure, which maintains the incompressibility of the fluid will be solved
each time step using a relaxational method.

The calculation will proceed like this:
    1. Set up the initial conditions and boundary conditions
    2. Update the pressure with a relaxational solver
    3. Update each velocity component
    4. Enforce boundary conditions
    5. Repeat 2-4 for every time-step
    
Here is depiction of the geometry of our flow:
            Wall with no flow.
        +-------------------------------------------------+
        |  --->                                           |     
        |                                                 |
        |  --->                                           |
Inflow  |                                                 | Outflow
        |  --->                                           |
        |                                                 |
        |  --->                                           |
        |                                                 |
        |  --->                                           |
        +-------------------------------------------------+
            Wall with no flow.
            
We expect our steady state solution to form a parabolic velocity curve.
Something similiar to this:

        +-------------------------------------------------+
        |  --->                         >                 |     
        |                               ->                |
        |  --->                         -->               |
Inflow  |                                                 | Outflow
        |  --->                         ----->            |
        |                                                 |
        |  --->                         -->               |
        |                               ->                |
        |  --->                         >                 |
        +-------------------------------------------------+
           
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm

# Define some simulation parameters
N_ITERATIONS = 3_000
REYNOLDS_NUMBER = 100

NX = 50 
NY = 30
DT = 0.001

LENGTH = 5
RADIUS = 1

NORM_TARGET = 1e-8

RHO = 1

INFLOW_VELOCITY = 0.04
PLOT_EVERY_N_STEPS = 100
PLOT_STEP_SKIP = 1000


def get_kinematic_viscosity(velocity, radius, reynolds_number):
    kinematic_viscosity = velocity * 2 * radius / reynolds_number
    return kinematic_viscosity


def pressure_solver(pressure, horizontal_velocity, vertical_velocity, 
                    rho, dt, dx, dy, norm_target):
    
    # We will define the following functions to compute the updated pressure
    # First we'll compute velocity dependent part of the pressure equation
    def get_velocity_dependent_part(horizontal_velocity, vertical_velocity, rho, dt, dx, dy, velocity_dependent_part):
        velocity_dependent_part = velocity_dependent_part.at[1:-1,1:-1].set(
            rho * (1 / dt * ((horizontal_velocity[1:-1,2:] - horizontal_velocity[1:-1,0:-2]) / (2 * dx) 
                              + (vertical_velocity[2:,1:-1] - vertical_velocity[0:-2, 1:-1]) / (2 * dy)) 
                              - ((horizontal_velocity[1:-1,2:] - horizontal_velocity[1:-1,0:-2]) / (2 * dx))**2          
                              - 2 * ((horizontal_velocity[2:,1:-1] - horizontal_velocity[0:-2, 1:-1] / (2 * dy)
                                      * (vertical_velocity[1:-1,2:] - vertical_velocity[1:-1,0:-2]) / (2 * dx))
                                      - ((vertical_velocity[2:,1:-1] - vertical_velocity[0:-2, 1:-1]) / (2 * dy)) **2))
            )
        return velocity_dependent_part
    
    def get_pressure_update(pressure, dx, dy, velocity_dependent_part):
        pressure_copy = pressure.copy()
        # Solve poisson's equation
        pressure = pressure.at[1:-1,1:-1].set((
            (pressure_copy[1:-1,2:] + pressure_copy[1:-1, 0:-2]) * dy**2 +
            (pressure_copy[2:,1:-1] + pressure_copy[0:-2,1:-1]) * dx**2) /
            (2 * (dx**2 + dy**2))- dx**2 * dy**2 / (2 * (dx**2 + dy**2)) *
            velocity_dependent_part[1:-1,1:-1])
        
        #update boundary conditions
        pressure =  pressure.at[:,-1].set(pressure[:,-2]) #no pressure change at outflow
    
        
        return pressure
    
    # Define loop init condition
    init = [pressure, pressure*0, horizontal_velocity, vertical_velocity, rho,
            dt, dx, dy, get_velocity_dependent_part(horizontal_velocity, vertical_velocity, rho, dt, dx, dy, pressure*0),
            norm_target]
    
    # We are setting up a JAX while loop, so we'll need a condition and body function
    def condition(loop_state):
        #Unpack Loop State
        pressure = loop_state[0]
        pressure_copy = loop_state[1]
        l1norm_target = loop_state[-1]
        
        ''' 
        Compute "norm" to compare the new pressure matrix to the previous one
        This solver uses a relaxational method. Each step makes finer adjustments
        to the pressure field. Once we reach an abritrarily small difference
        between each step, we send the signal to terminate the loop.
        '''
        norm = (jnp.sum(jnp.abs(pressure[:])-jnp.abs(pressure_copy[:])) / jnp.sum(jnp.abs(pressure_copy[:])))
        return norm > l1norm_target
    
    
    def body(loop_state):
        #unpack loop_state
        pressure, pressure_copy, horizontal_velocity, vertical_velocity, rho, dt, dx, dy, velocity_dependent_part, norm_target = loop_state
        # compute pressure update
        pressure = get_pressure_update(pressure, dx, dy, velocity_dependent_part)
        # repack loop_state
        loop_state = [pressure, pressure_copy, horizontal_velocity, vertical_velocity, rho, dt, dx, dy, velocity_dependent_part, norm_target]
        return loop_state
    # Loop pressure steps until pressure has relaxed to desired degree
    result = jax.lax.while_loop(condition,
                                body,
                                init)
    
    return result[0]

# Now we need to compute the velocity updates
@jax.jit
def horizonal_velocity_update(horizontal_velocity, vertical_velocity, rho, dt, dx, dy, pressure, kinematic_viscosity):
    horizontal_velocity_copy = horizontal_velocity.copy()
    vertical_velocity_copy = vertical_velocity.copy()
    horizontal_velocity = horizontal_velocity.at[1:-1,1:-1].set(
        (horizontal_velocity_copy[1:-1,1:-1] - 
         horizontal_velocity_copy[1:-1,1:-1] * dt / dx * 
         (horizontal_velocity_copy[1:-1,1:-1] - horizontal_velocity_copy[1:-1, 0:-2]) -
         vertical_velocity_copy[1:-1,1:-1] * dt / dy * 
         (horizontal_velocity_copy[1:-1,1:-1] - horizontal_velocity_copy[0:-2,1:-1]) -
         dt / (2 * rho * dx) * (pressure[1:-1, 2:] - pressure[1:-1,0:-2]) +
         kinematic_viscosity * (dt / dx ** 2 *
        (horizontal_velocity_copy[1:-1,2:] - 2 * horizontal_velocity_copy[1:-1, 1:-1] +
         horizontal_velocity_copy[1:-1, 0:-2]) + dt / dy**2 *
        (horizontal_velocity_copy[2:,1:-1] - 2 * horizontal_velocity_copy[1:-1,1:-1] +
         horizontal_velocity_copy[0:-2, 1:-1])))
        )
    
    return horizontal_velocity
@jax.jit
def vertical_velocity_update(horizontal_velocity, vertical_velocity, rho, dt, dx, dy, pressure, kinematic_viscosity):
    horizontal_velocity_copy = horizontal_velocity.copy()
    vertical_velocity_copy = vertical_velocity.copy()
    
    horizontal_velocity = horizontal_velocity.at[1:-1,1:-1].set(
        (vertical_velocity_copy[1:-1,1:-1] - 
         horizontal_velocity_copy[1:-1,1:-1] * dt / dx * 
         (vertical_velocity_copy[1:-1,1:-1] - vertical_velocity_copy[1:-1, 0:-2]) -
         vertical_velocity_copy[1:-1,1:-1] * dt / dy * 
         (vertical_velocity_copy[1:-1,1:-1] - vertical_velocity_copy[0:-2,1:-1]) -
         dt / (2 * rho * dx) * (pressure[2:, 1:-1] - pressure[0:-2,1:-1]) +
         kinematic_viscosity * (dt / dx** 2 *
        (vertical_velocity_copy[1:-1,2:] - 2 * vertical_velocity_copy[1:-1, 1:-1] +
         vertical_velocity_copy[1:-1, 0:-2]) + dt / dy**2 *
        (vertical_velocity_copy[2:,1:-1] - 2 * vertical_velocity_copy[1:-1,1:-1] +
         vertical_velocity_copy[0:-2, 1:-1])))
        )
    
    return vertical_velocity

def main():
    jax.config.update("jax_enable_x64", True)
    
    kinematic_viscosity = get_kinematic_viscosity(INFLOW_VELOCITY, RADIUS, REYNOLDS_NUMBER)
    
    x = jnp.arange(NX)
    y = jnp.arange(NY)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    dx = 2 / (NX - 1)
    dy = 2 / (NY - 1)
    
    # Initialize variable matricies 
    
    pressure = jnp.zeros((NX,NY))
    horizontal_velocity = jnp.zeros((NX,NY))
    vertical_velocity = jnp.zeros((NX,NY))
    
    # Boundary Conditions
    
    for iteration_index in tqdm(range(N_ITERATIONS)):
        
        pressure = pressure_solver(pressure, horizontal_velocity, vertical_velocity, RHO, DT, dx, dy, NORM_TARGET)
        horizontal_velocity = horizonal_velocity_update(horizontal_velocity, vertical_velocity, RHO, DT, dx, dy, pressure, kinematic_viscosity)
        vertical_velocity = vertical_velocity_update(horizontal_velocity, vertical_velocity, RHO, DT, dx, dy, pressure, kinematic_viscosity)
        
if __name__=="__main__":
    main()