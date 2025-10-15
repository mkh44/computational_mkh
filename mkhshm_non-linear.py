#!/usr/bin/env python3
from stringprep import b1_set

# PHY1063 Computational Physics
#
# Verifying a solution of the simple harmonic oscillator equation by
# establishing convergence to an exact solution.

import matplotlib.pyplot as plt
import numpy as np

k      = 1.0 # Spring constant
m      = 1.0 # Mass
cycles = 2 # No. of periods to integrate over
x0     = 1.0 # Initial displacement
v0     = 0.0 # Initial velocity
alpha = 0.1 #nonlinear spring parameter (alpha = 0 will be for pure SMH)
omega = (k/m)**0.5

# Special case variables
target_omega_delta = 0.4
total_time = 2.0 * cycles * np.pi / omega
delta = target_omega_delta / omega


def leapfrog_nonlinear(steps):
    delta = total_time /steps
    t = np.empty( steps + 1 )
    x = np.empty( steps + 1)
    v = np.empty(steps + 1)
    t[0] = 0.0
    x[0] = x0
    v[0] = 0.0
    #inital half-step velocity
    a0 = (-k * x[0] - alpha * x[0]**3)/m
    v[0] = v0 + 0.5 * delta * a0

    for i in range(steps):
        #advancing position by full step using half-step vel
        a = (-k * x[i] - alpha * x[i]**3)/m
        x[i+1] = x[i] + delta * v[i]
        t[i+1] = t[i] + delta

        # new acceleration
        a_new = (-k * x[i+1] - alpha * x[i+1]**3)/m
        v[i + 1] = v[i] + delta * a_new


    return t, x, v


# This loop integrates the SHM equations repeatedly using an increasing
# number of steps (doubling at each loop iteration).
n        = 14
steps    = 8
delta    = np.empty( n )
solutions = []
time_arrays = []

for i in range(0,n):
    t, x, v     = leapfrog_nonlinear( steps )
    solutions.append(x)
    time_arrays.append(t)
    delta[i]    = omega*(t[1]-t[0])
    plt.plot( t, x, label=f"steps={steps}")
    plt.ylabel("x(t)")
    plt.xlabel("t")
    plt.legend(loc = "best")
    steps *= 2

plt.show()

# use the finest time grid as reference (last one)
t_ref = time_arrays[-1]
x_ref = solutions[-1]

plt.figure(figsize=(10,6))
for i in range(len(solutions)-1):
    t = time_arrays[i]
    x = solutions[i]
    x_interp = np.interp(t_ref, t, x)
    diff = x_interp - x_ref
    plt.plot(t_ref, diff, label=f"Δx for steps={len(t)-1}")

plt.xlabel("t")
plt.ylabel("x_difference(t)")
plt.title("Difference Between Numerical Solutions")
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.show()

steps = 8 * 2**(n-1)   # same as last loop iteration
t, x, v = leapfrog_nonlinear(steps)

# energies
kin = 0.5 * m * v**2
pot = 0.5 * k * x**2 + 0.25 * alpha * x**4
tot = kin + pot

#plotting special case
plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(t, x, label="x(t)", color='purple')
plt.ylabel("x")
plt.legend(loc="best")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t, v, label="v(t)", color='tab:red')
plt.ylabel("v")
plt.legend(loc="best")
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(t, kin, label='Kinetic', alpha=0.9)
plt.plot(t, pot, label='Potential', alpha=0.9)
plt.plot(t, tot, label='Total', linewidth=1.5, color='k')
plt.xlabel("t")
plt.ylabel('Energy')
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()



