# Hybrid-BHE-Simulation-Model

This repository contains a hybrid simulation approach for 1U-,
2U- and coaxial single, and field of, borehole heat exchangers.
We implement a novel combination of existing solutions for
the simulation of heat transfer processes within the borehole
and the surrounding ground. The heat transfer in the ground
is modelled with a combination of analytically determined
g-functions, the borehole models utilize thermal resistance
capacity models and the finite volume method. Critically, we
improve the computational efficiency of long-term simulations
by sub-dividing the time scale into multiple periods, where the
influence of past periods on future periods is calculated using
the Fast Fourier Transform. 
