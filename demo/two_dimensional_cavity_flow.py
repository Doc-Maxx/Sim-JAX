"""
@author: Doc-Maxx

This code will simulate two dimensional flow in a cavity.

The initial and boundary conditions are from Zhengtao Gan's online CFD course.
The results from that will be used to validate the results here.
"""
import jax
import jax.numpy as jnp
from jax import jit
from matplotlib import pyplot as plt, cm
import cmasher as cmr
from tqdm import tqdm

# Define some simulation parameters
N_ITERATIONS = 500
REYNOLDS_NUMBER = 100

NX = 41 
NY = 41
DT = 0.001

LENGTH = 2
RADIUS = 2

NORM_TARGET = 1e-8
RHO = 1

C = 1
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
        #set inlet pressure
        pressure = pressure.at[:, -1].set(pressure[:, -2]) # dp/dx = 0 at x = 2
        pressure = pressure.at[0,:].set(pressure[1, :])   # dp/dy = 0 at y = 0
        pressure = pressure.at[:,0].set(pressure[:, 1])   # dp/dx = 0 at x = 0
        pressure = pressure.at[-1,:].set(0)        # p = 0 at y = 2
        #jax.debug.print("{x}", x =  pressure )
        return pressure, pressure_copy
    
    # Define loop init condition
    init = [pressure, pressure+1, horizontal_velocity, vertical_velocity, rho,
            dt, dx, dy, pressure, norm_target]
    
    # We are setting up a JAX while loop, so we'll need a condition and body function
    def condition(loop_state):
        #Unpack Loop State
        pressure = loop_state[0]
        pressure_copy = loop_state[1]
        l1norm_target = loop_state[-1]
        
        ''' 
        Compute "norm" to compare the new pressure matrix to the previous one
        This solver uses a relaxational method. Each step makes finer adjustments|
        to the pressure field. Once we reach an abritrarily small difference
        between each step, we send the signal to terminate the loop.
        '''
        norm = (jnp.sum(jnp.abs(pressure[:]-pressure_copy[:])) / (jnp.sum(jnp.abs(pressure_copy[:]))+1e-8 ))
        #jax.debug.print("{x}", x=norm)
        return norm > l1norm_target
    
    
    def body(loop_state):
        #unpack loop_state
        pressure, pressure_copy, horizontal_velocity, vertical_velocity, rho, dt, dx, dy, velocity_dependent_part, norm_target = loop_state
        # compute pressure update
        velocity_dependent_part = get_velocity_dependent_part(horizontal_velocity, vertical_velocity, rho, dt, dx, dy, velocity_dependent_part)
        #jax.debug.print("{x}", x = velocity_dependent_part)
        pressure, pressure_copy = get_pressure_update(pressure, dx, dy, velocity_dependent_part)
        #jax.debug.print("Pressure body loop : {x}", x =  pressure )
        # repack loop_state
        loop_state = [pressure, pressure_copy, horizontal_velocity, vertical_velocity, rho, dt, dx, dy, velocity_dependent_part, norm_target]
        return loop_state
    # Loop pressure steps until pressure has relaxed to desired degree
    # result has the same structure of loop_state
    result = jax.lax.while_loop(condition,
                                body,
                                init)
    # returning only the pressure.
    return result[0]

# Now we need to compute the velocity updates
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

def vertical_velocity_update(horizontal_velocity, vertical_velocity, rho, dt, dx, dy, pressure, kinematic_viscosity):
    horizontal_velocity_copy = horizontal_velocity.copy()
    vertical_velocity_copy = vertical_velocity.copy()
    
    vertical_velocity = vertical_velocity.at[1:-1,1:-1].set(
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
@jit
def main():
    jax.config.update("jax_enable_x64", True)
    
    kinematic_viscosity = 0.001
    
    x = jnp.linspace(0, LENGTH, NX)
    y = jnp.linspace(0, RADIUS, NY)
    
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    dx = LENGTH / (NX - 1)
    dy = RADIUS / (NY - 1)
    
    # Initialize variable matricies 
    
    pressure = jnp.zeros((NX,NY))
    horizontal_velocity = jnp.zeros((NX,NY))
    vertical_velocity = jnp.zeros((NX,NY))
    
    #set inlet pressure
    pressure = pressure.at[:, -1].set(pressure[:, -2]) # dp/dx = 0 at x = 2
    pressure = pressure.at[0,:].set(pressure[1, :])   # dp/dy = 0 at y = 0
    pressure = pressure.at[:,0].set(pressure[:, 1])   # dp/dx = 0 at x = 0
    pressure = pressure.at[-1,:].set(0)        # p = 0 at y = 2
    
    horizontal_velocity = horizontal_velocity.at[0, :].set(0)
    horizontal_velocity = horizontal_velocity.at[:, 0].set(0)
    horizontal_velocity = horizontal_velocity.at[:, -1].set(0)
    horizontal_velocity = horizontal_velocity.at[-1, :].set(C)    # set velocity on cavity lid equal to C
    vertical_velocity = vertical_velocity.at[0, :].set(0)
    vertical_velocity = vertical_velocity.at[-1, :].set(0)
    vertical_velocity = vertical_velocity.at[:, 0].set(0)
    vertical_velocity = vertical_velocity.at[:, -1].set(0)
    
    pressure = pressure_solver(pressure, horizontal_velocity, vertical_velocity, RHO, DT, dx, dy, NORM_TARGET)

    for iteration_index in tqdm(range(N_ITERATIONS)):
        
        
        horizontal_velocity = horizonal_velocity_update(horizontal_velocity, vertical_velocity, RHO, DT, dx, dy, pressure, kinematic_viscosity)
        vertical_velocity = vertical_velocity_update(horizontal_velocity, vertical_velocity, RHO, DT, dx, dy, pressure, kinematic_viscosity)
        
        # Boundary Conditions
        horizontal_velocity = horizontal_velocity.at[0, :].set(0)
        horizontal_velocity = horizontal_velocity.at[:, 0].set(0)
        horizontal_velocity = horizontal_velocity.at[:, -1].set(0)
        horizontal_velocity = horizontal_velocity.at[-1, :].set(C)    # set velocity on cavity lid equal to C
        vertical_velocity = vertical_velocity.at[0, :].set(0)
        vertical_velocity = vertical_velocity.at[-1, :].set(0)
        vertical_velocity = vertical_velocity.at[:, 0].set(0)
        vertical_velocity = vertical_velocity.at[:, -1].set(0)
        
        pressure = pressure_solver(pressure, horizontal_velocity, vertical_velocity, RHO, DT, dx, dy, NORM_TARGET)
        #jax.debug.print("{x}", x =  vertical_velocity )
        
    # Create figure and set dpi and figure size
    fig = plt.figure(figsize=(11,7), dpi=100)
    
    # Contourf plot for pressure field with colorbar
    cf = plt.contourf(X, Y, pressure, alpha=0.5, cmap='turbo', levels=10)
    plt.colorbar(cf, label='Pressure')
    
    # Contour plot for pressure field outlines
    contour = plt.contour(X, Y, pressure, cmap='turbo', levels=10)
    plt.clabel(contour, inline=False, fontsize=12, colors = 'black')
    
    # Quiver plot for velocity field
    quiv = plt.quiver(X[::2, ::2], Y[::2, ::2], 
                      vertical_velocity[::2, ::2], 
                      horizontal_velocity[::2, ::2]) 
    
    # Setting labels for the x and y axes
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    
    # Setting the title for the plot
    plt.title('Pressure and Velocity fields', fontsize=14)
    
    # Display the plot
    plt.show()
            
    
if __name__=="__main__":
    main()