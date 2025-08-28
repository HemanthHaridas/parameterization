## Particle Swarm Optimization Implementation for FF Parameterization

<p align="justify"> A Python implementation of a Particle Swarm Optimization (PSO) algorithm for parameter optimization and tuning. </p>

### Overview
<p align="justify"> This repository contains a robust implementation of the Particle Swarm Optimization (PSO) algorithm. PSO is a computational method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality. The algorithm is inspired by the social behavior of bird flocking or fish schooling. </p>

<p align="justify"> The multiple walker approach enhances the traditional PSO algorithm by iteratively generating new walkers within the vicinity of the best walker found in the previous generation, to explore the solution space simultaneously, improving convergence and helping avoid local optima. </p>

<p align="justify"> The code is fully parallelized using OpenMP routines provided by mpi4py package, and will spawn atmost nproc number of walkers, based on the number of processors requested.