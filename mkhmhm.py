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
cycles = 2.0 # No. of periods to integrate over
x0     = 1.0 # Initial displacement
v0     = 0.0 # Initial velocity
alpha = 0.0 #nonlinear spring parameter (alpha = 0 will be for pure SMH)
omega = (k/m)**0.5

def leapfrog( steps ):
    """Solve the simple harmonic motion equations for several oscillation cycles,
       assuming that the mass (m) and spring constant (k) are defined in the
       global space.
    """

    delta = (2.0*cycles*np.pi/omega)/steps
    x     = np.empty( steps+1 )
    v     = np.empty( steps+1 )
    t     = np.empty( steps+1 )
    t[0]  = 0.0
    x[0]  = x0
    #Initial half-step velocity
    a0 = (-k * x[0] - alpha * x[0]**3) / m
    v[0]  = v0 + 0.5 * delta * a0

    for i in range(steps):
        #Acceleration from modified force law
        #a = (-k * x[i] - alpha * x[i]**3) / m
        a = (-k * x[i] - alpha * x[i] ** 3) / m
        v[i + 1] = v[i] + delta * a
        t[i+1] = t[i] + delta
        x[i+1] = x[i] + delta*v[i+1]
    return t, x, v

def l2_error_norm(t, x):
    """Calculate the L2 relative error norm."""
    steps = len( x ) - 1
    l2_err = 0.0
    l2     = 0.0
    l2_ref = 0.0
    for i in range(len(x)):
        x_exact = x0*np.cos( omega*t[i] )
        l2_err += ((x[i] - x_exact)**2.0)
        l2     += (x[i]**2.0)
        l2_ref += (x_exact**2.0)
    return (l2_err/l2)**0.5

# The backend choice here may be platform dependent. You may need to
# change 'TkAgg' to something else (or omit this line entirely).

plt.switch_backend( 'TkAgg' )

# This loop integrates the SHM equations repeatedly using an increasing
# number of steps (doubling at each loop iteration).
n        = 14
steps    = 8
l2_error = np.empty( n )
delta    = np.empty( n )

for i in range(n):
    t, x, v     = leapfrog( steps )
    delta[i]    = omega*(t[1]-t[0])
    l2_error[i] = l2_error_norm( t , x )
    plt.plot( t, x, label=f"steps={steps}")
    plt.ylabel("x(t)")
    plt.xlabel("t")
    plt.legend(loc = "best")
    #plt.show( block=False )
    steps *= 2

#overlaying exact solution on to fig 1
t_dense = np.linspace(0, 2.0*cycles*np.pi/omega, 1000)
x_exact = x0 * np.cos(omega * t_dense)
plt.plot(t_dense, x_exact, 'k--', linewidth=2, label='Exact solution')
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Numerical solutions vs manufactured exact solution")
plt.legend(fontsize=8)
plt.grid(True, linestyle=':')
plt.tight_layout()

# Switch to a new plotting window, and plot the L2 error norm,
# with guidelines for first, second, and third order accuracy.
plt.figure()
plt.loglog( delta , l2_error, 'o', label="L2 Error numerical results" )
plt.loglog( delta , l2_error[0]*(delta/delta[0])**1.0, label='1st order accuracy' )
plt.loglog( delta , l2_error[0]*(delta/delta[0])**2.0, label='2nd order accuracy' )
plt.loglog( delta , l2_error[0]*(delta/delta[0])**3.0, label='3rd order accuracy' )

#calculating convergence rate (slope of log.log)
fine_region = slice(-14, None) #using smallest dt values for better acuracy
fit = np.polyfit(np.log(delta[fine_region]), np.log(l2_error[fine_region]), 1)
slope = fit[0]

fit_line = np.exp(fit[1]) *delta**slope
plt.loglog(delta, fit_line, '--', label=f'Fitted slope = {slope:3f}')


plt.xlabel('Δt x ω')
plt.ylabel('Relative Error Norm')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()

plt.show()
