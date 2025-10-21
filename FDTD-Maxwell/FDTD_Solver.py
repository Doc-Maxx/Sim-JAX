#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 13:44:14 2025

@author: Doc-Maxx
"""

import jax
import jax.numpy as jnp
from tqdm import tqdm

def update_Hx(Hx, Ez, dt, dx):
    Hx = Hx.at[1:-1,1:-1].set(
        Hx[1:-1,1:-1] - dt / dy * (Ez[:-2, 1:-1] - Ez[2:, 1:-1])
        )  
    return Hx

def update_Hx(Hy, Ez, dt, dx):
    Hy = Hy.at[1:-1,1:-1].set(
        Hy[1:-1,1:-1] - dt / dx * (Ez[1:-1,:-2] - Ez[1:-1,2:])
        )  
    return Hx

def update_Ez(Ez, Hx, Hy, Eps_i, Eps_r, omega, dt, dx):
    Ez = Ez.at[1:-1,1:-1].set(
        Ez[1:-1,1:-1]*jnp.exp(- Eps_i[1:-1,1:-1] * omega * dt / Eps_r[1:-1,1:-1])
        + c**2 * dt / Eps_r[1:-1,1:-1] * ( 1 / dx * (Hy[1:-1,1:-1] - Hy[1:-1,:-2])
        +  1 / dy * (-Hx[1:-1,1:-1] + Hx[2:, 1:-1]) )
        )
    return Ez

