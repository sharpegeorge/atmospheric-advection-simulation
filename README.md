# atmospheric-advection-simulation

### Overview

This project numerically solves the 2D advection equation to simulate the movement of a cloud of material in the Atmospheric Boundary Layer. It includes domain scaling, velocity shear using a logarithmic wind profile, and OpenMP parallelisation.

### Features

- Finite difference method with forward Euler time stepping to solve the 2D advection equation
- Key computational loops parallelised using OpenMP to improve performance on multi-core systems
- Vertical wind shear model using a logarithmic velocity profile to represent the atmospheric boundary layer
- Calculates the vertically averaged concentration distribution to assess horizontal plume spread
- Generates .dat files suitable for direct plotting with Gnuplot or further analysis with other tools

### How to run
1. Clone the repository
   ```bash
   git clone https://github.com/sharpegeorge/atmospheric-advection-simulation.git
   cd atmospheric-advection-simulation
   ```

2. Compile the Serial Version
   ```bash
   gcc -o main -std=c99 main.c -lm
   ```

4. Compile the OpenMP Version
   ```bash
   gcc -fopenmp -o main -std=c99 main.c -lm
   ```

5. Run the simulation
   ```bash
   ./main
   ```
   
### Files Included
- `main.c`
- `plottingScript`
