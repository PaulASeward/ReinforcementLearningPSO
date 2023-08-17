from Standard_PSO import *
from Vectorized_PSOGlobalLocal import *
import time

# Fitness of the global optima for the 28 functions of the CEC'13 benchmark
fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]


def run_experiment(runs, dimensions, evals_per_dim, swarm_size, global_topology, vectorized):
    function_result_array = []
    run_results = []
    obj_f = functions.CEC_functions(dimensions)
    topology_label = (lambda: 'Global' if global_topology else 'Local')()
    comp_label = (lambda: 'Vectorized' if vectorized else 'Iterative')()

    for fun_num in range(1, 10):
        error_list = []  # Store errors for each run of the current function
        run_result = []
        total_time = 0.0

        for run in range(1, runs+1):
            # print(f"Function {fun_num}. Run Number: {run}/{runs}. Topology: {topology_label}. Vectorized: {vectorized}")

            if vectorized:
                pso = PSOVectorSwarmGlobalLocal(obj_f, fun_num, dimensions, evals_per_dim, swarm_size, global_topology=global_topology)
            else:  # Standard Approach (Non-Vectorized)
                pso = ParticleSwarmOptimizer(fun_num)

            start_time = time.time()  # Start timing
            gbest_val, gbest_pos = pso.optimize()
            end_time = time.time()  # End timing

            run_time = end_time - start_time  # Calculate the time taken for this run
            error = gbest_val - fDeltas[fun_num - 1]  # Calculate the error for this run.

            error_list.append(error)  # Add error to the list
            run_result.append([fun_num, run, error, run_time])  # Store result for this run

            total_time += run_time  # Accumulate the total time

            # print(f"Current result (error respect to the global optimum): {error:.2E}")
            # print(f"Time taken for this run: {run_time:.2f} seconds")

        avg_error = np.sum(error_list) / runs
        avg_time = total_time / runs
        std_dev = np.std(error_list)  # Calculate standard deviation

        # print(f"Function {fun_num}, result (Average error respect to the global optimum): {avg_error:.2E}")
        # print(f"Average time taken: {avg_time:.2f} seconds")
        run_results.append(run_result)
        function_result_array.append([fun_num, avg_error, std_dev, avg_time])  # Include std_dev in the result array

    # Save run results to a CSV file
    with open(f'{topology_label}{comp_label}runresults.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Function Number', 'Run Number', 'Error', 'Time'])  # Header for run results
        for run_result in run_results:
            writer.writerows(run_result)  # Write each run result

    # Save function number summative results to a CSV file
    with open(f'{topology_label}{comp_label}functionresults.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Function Number', 'Mean Error', 'Standard Deviation', 'Mean Time'])  # Header for result array
        writer.writerows(function_result_array)  # Write result array


runs = 30
dimensions = 30
swarm_size = 50
evals_per_dim = 10000
run_experiment(runs, dimensions, evals_per_dim, swarm_size, global_topology=True, vectorized=True)
run_experiment(runs, dimensions, evals_per_dim, swarm_size, global_topology=False, vectorized=True)
run_experiment(runs, dimensions, evals_per_dim, swarm_size, global_topology=False, vectorized=False)
