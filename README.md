# Vectorized brute force gravitational N-body simulation

We take advantage of _numpy_'s broadcasting capabilities to vectorize a simple graviational brute force N-body simulation. We then employ this method in the study of astrophysical systems.

Code is meant to be simple and readable. Everything bellow #Application is an application of the code to generate data for given article.

Everything in init_cond generates samples of initial conditions and the integrator integrates the system for given time intervals with given time step.
