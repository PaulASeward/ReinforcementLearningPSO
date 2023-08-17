from Standard_PSO import *
from Vectorized_PSOGlobalLocal import *
# from functions import *
import time

# Fitness of the global optima for the 28 functions of the CEC'13 benchmark
fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

def run_experiment(runs, dimensions, evals_per_dim, swarm_size, global_topology, vectorized):
    obj_f = functions.CEC_functions(dimensions)

    topology_label = 'Global' if global_topology else 'Local'
    comp_label = 'Vectorized' if vectorized else 'Iterative'

    # Save run results to a CSV file
    with open(f'{topology_label}{comp_label}runresults.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Function Number', 'Run Number', 'Error', 'Time'])  # Header for run results

    with open(f'{topology_label}{comp_label}functionresults.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Function Number', 'Mean Error', 'Standard Deviation', 'Mean Time'])  # Header for result array

    for fun_num in range(1, 29):
        error_list = []
        run_results = []
        total_time = 0.0

        for run in range(1, runs+1):
            # print(f"Function {fun_num}. Run Number: {run}/{runs}. Topology: {topology_label}. Vectorized: {vectorized}")
            if vectorized:
                pso = PSOVectorSwarmGlobalLocal(obj_f, fun_num, dimensions, evals_per_dim, swarm_size, global_topology=global_topology)
            else:
                pso = ParticleSwarmOptimizer(fun_num)

            start_time = time.time()
            gbest_val, gbest_pos = pso.optimize()
            end_time = time.time()

            run_time = end_time - start_time
            error = gbest_val - fDeltas[fun_num - 1]

            error_list.append(error)
            run_results.append([fun_num, run, error, run_time])

            total_time += run_time

            # Write current result to CSV file
            with open(f'{topology_label}{comp_label}runresults.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([fun_num, run, error, run_time])

        avg_error = np.sum(error_list) / runs
        avg_time = total_time / runs
        std_dev = np.std(error_list)

        function_result_array = [[fun_num, avg_error, std_dev, avg_time]]
        # Write function number summative result to CSV file
        with open(f'{topology_label}{comp_label}functionresults.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([fun_num, avg_error, std_dev, avg_time])

runs = 30
dimensions = 30
swarm_size = 50
evals_per_dim = 10000
run_experiment(runs, dimensions, evals_per_dim, swarm_size, global_topology=True, vectorized=True)
run_experiment(runs, dimensions, evals_per_dim, swarm_size, global_topology=False, vectorized=True)
run_experiment(runs, dimensions, evals_per_dim, swarm_size, global_topology=False, vectorized=False)
