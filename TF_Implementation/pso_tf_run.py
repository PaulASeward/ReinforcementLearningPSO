from PSOtf import *
from functions import *
import time

# Fitness of the global optima for the 28 functions of the CEC'13 benchmark
fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]


def run_experiment(runs, dimensions, evals_per_dim, swarm_size):
    obj_f_tf = functionstf.CEC_functions(dimensions)

    fun_num=19

    pso = PSOtf(obj_f_tf, fun_num, dimensions, evals_per_dim, swarm_size)

    start_time = time.time()
    pso.optimize()
    end_time = time.time()
    run_time = end_time - start_time

    gbest_val = pso.get_current_best_fitness()
    error = gbest_val - fDeltas[fun_num - 1]

    # Print the gbest_val and error
    print(f"Current Best Fitness (gbest_val): {gbest_val}")
    print(f"Error: {error}")
    print(f"Time: {run_time}")

runs = 1
dimensions = 30
swarm_size = 50
evals_per_dim = 10000
run_experiment(runs, dimensions, evals_per_dim, swarm_size)
