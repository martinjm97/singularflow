# SingularFlow
This repository contains the implementation of **Semantics of Integrating and Differentiating Singularities**.

For background, a *singular function* is a partial function such that at one or more points, the left and/or right limit diverge (e.g., the function \(1/x\)). Since programming languages typically support division, programs may denote singular functions. Although on its own, a singularity may be considered a bug, introducing a division-by-zero error, *singular integrals*—a version of the integral that is well-defined when the integrand is a singular function and the domain of integration contains a singularity—arise naturally in science and engineering, including physics, aerodynamics, mechanical engineering, and computer graphics.

The core contributions that this artifact supports are:

(1) We implement **SingularFlow** in JAX and evaluate the implementation on a suite of benchmarks that perform the *finite Hilbert transform*, an integral transform similar to the Fourier transform, that arises in domains such as physics and electrical engineering. 

(2) We use **SingularFlow** to approximate the solutions to four *singular integral equations*—equations where the unknown function is in the integrand of a singular integral—arising in aerodynamics and mechanical engineering.


## Getting started guide
SingularFlow extends JAX with support for singular integration and the evaluation requires only commonly used dependencies. 
Nevertheless, we start by providing a guide to getting the environment set up. 

### Setup
We briefly describe how to install the relevant dependencies.

#### Install a recent version of Python
First we get a recent version of python up and running (e.g., python 3.12). 
One way to accomplish this is to [install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and then create a new environment:

`conda create -n singularflow python=3.12`,

and activate it with:

`conda activate singularflow`.

#### Installing requirements
Clone the `SingularFlow` repository and go into the cloned directory with `cd singularflow`.
You can install all requirements by running:

`pip install -r requirements.txt`,

where the primary dependencies are `flax`, `jax`, `matplotlib`, `pandas`, and `typed-argument-parser`.

### Quick test
To check that all the dependencies are correctly installed, just run:

`python hilbert.py --table1 --table2`

and

`python airfoil.py --run True`.

For the above commands outputs should be (1) two tables looking like: 
```
      f(u)  Ground Truth  Ours    ...   
0       u        -0.461 -0.46 
1     u^2        -0.231 -0.23 
...
```
and (2) output that looks like:
```
Iteration 0: train loss 6.8445106 test loss 6.9290433
Iteration 1: train loss 5.725188 test loss 5.202486
.
.
.
Iteration 999: train loss 0.0016780817 test loss 0.07308424
Training time: 16.773061275482178 seconds
```
Each of these commands should run in <30 seconds.

## Code overview

Code layout:

`singular_integrate.py`: Provides the core implementation that extends JAX with the ability to differentiate singular integrals.

`airfoil.py`: The script for solving the airfoil equation using a neural network (Figures 2 and 9).

`lem_starr.py`: The script for solving the 1D crack problem using a neural network (Figure 10). LEM stands for linear elastic mechanics, which is the subarea of mechanical engineering that studies this problem.

`lem_pr1.py`: The script for solving the 2D crack problem using a neural network (Figure 11).

`hilbert.py`: The implementation of the Hilbert transform, it's derivative and the timing results. This corresponds to Tables 1-4.

`data`: Outputs of runs go to the `data` folder. Runs including neural network training produce multiple files in a folder named with the current date and the name of the experiment (e.g., `data/march_13_2025/airfoil/`)

`training.py`: The core neural network training code used in the lem and airfoil scripts.

`plotting.py`: The plotting scripts used in the lem and airfoil scripts.

`airfoil_plot.py`: Produces Figure 2a (with flag `--plot_chord True`) and Figure 9a (with flags `--max_camber 0.00 --location_of_max_camber 0.0 --max_thickness 0.12`).

`crack_plot.py`: Produces the crack visualization in Figure 11a.

`utils.py`: Some helper functions, primarily for `hilbert.py`.

## Step-by-step instructions for reproducing experimental results

Below are detailed instructions for reproducing the core experimental results in the paper. 

### Finite Hilbert transform results
The finite Hilbert transform is a singular integral transform used in signal processing and physics. 
The finite Hilbert transform of a function `f(u)`, defined as:

`g(s) = -1/pi (singular integral_a^b f(u) / (u - s) dx`.

We study the case, where `a=-1` and `b=1`.

#### Reproducing the finite Hilbert Transform (Table 1)

Table 1 depicts the comparison of a number of techniques for calculating the finite Hilbert transform `g(s)` for different functions. 
You can reproduce the results by running:

`python hilbert.py --table1`.

#### Reproducing the derivative of the finite Hilbert transform (Table 2)

Table 2 depicts the comparison of a number of techniques for calculating the derivative of the finite Hilbert transform `g(s)` for different functions. 
You can reproduce the results by running:

`python hilbert.py --table2`.

#### Reproducing timing results (Table 3 and Appendix D.1)

While precise reproduction of timing results is not possible, results should not be too different on similar hardware. We used a 2019 Intel Macbook Pro with an Intel Core i9 processor and 32GB of RAM for the results in the paper. 

To (approximately) reproduce the results in Table 3, run:

`python hilbert.py --table3`.

To (approximately) reproduce the results in Table 4 in Appendix D.1, run:

`python hilbert.py --appx_computation_time`.


### Solving singular integral equations using neural networks

We use the physically-informed neural network (PINN) approach to solving singular integral equations using neural networks.
The key idea is to take an integral of the form: 

`singular integral f(x) / (x - s)^k dx = g(s)`,

to replace `f(x)` by a neural network `nn(x, theta)` and find the `theta` that minimizes the loss:

`(singular integral nn(x, theta) / (x - s)^k dx - g(s))`.

Check results in the paper by loading neural networks that approximate solutions singular integral equations and plotting the training and test losses as well as the function that the neural network represents.
Alternatively, you can train the neural networks from scratch. The current settings should precisely reproduce all plots in the paper.


#### Load results from the paper

It should take only a few seconds to load the data and plot the results.

To reproduce the loss plots and function plot for Figure 2:

`python airfoil.py --plot True --load_files AllSeeds --num_seeds 5 --path march_13_2025`

To reproduce the loss plots and function plot for Figure 9:

`python airfoil.py --plot True --load_files AllSeeds --num_seeds 5 --path march_13_2025 --max_camber 0.00 --location_of_max_camber 0.0 --max_thickness 0.12`

To reproduce the loss plots and function plot for Figure 10:

`python lem_starr.py --plot True --load_files AllSeeds --num_seeds 5 --path march_13_2025`

To reproduce the loss plots and function plot for Figure 11:

`python lem_pr1.py --plot True --load_files AllSeeds --num_seeds 5 --path march_13_2025`

#### Training neural networks to solve singular integral equations from scratch

It should take around 20 seconds to train any of these models from scratch.

To reproduce the training run from Figure 2:

`python airfoil.py --run True --save_files AllSeeds --num_seeds 5`

To reproduce the training run from Figure 9:

`python airfoil.py --run True --save_files AllSeeds --num_seeds 5 --max_camber 0.00 --location_of_max_camber 0.0 --max_thickness 0.12`

To reproduce the training run from Figure 10:

`python lem_starr.py --run True --save_files AllSeeds --num_seeds 5`

To reproduce the training run from Figure 11:

`python lem_pr1.py --save_files AllSeeds --run True --num_seeds 5`



