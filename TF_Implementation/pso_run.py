import functions
from Vectorized_PSOGlobalLocal import *
import time
import pandas as pd

# Fitness of the global optima for the 28 functions of the CEC'13 benchmark
fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

def run_experiment(runs, dimensions, evals_per_dim, swarm_size):
    obj_f = functions.CEC_functions(dimensions)
    fun_num=19

    pso = PSOVectorSwarmGlobalLocal(obj_f, fun_num, dimensions, evals_per_dim, swarm_size)
    start_time = time.time()
    pso.optimize()
    end_time = time.time()
    run_time = end_time - start_time

    observation_array1, observation_array2, observation_array3 = pso.get_observation()
    print(observation_array1.shape)
    print(observation_array2.shape)
    print(observation_array3.shape)

    _observation = np.concatenate([observation_array1, observation_array2, observation_array3], axis=0)
    print(_observation.shape)


    V, relative_fitness, pbest_replacement_batchcounts = pso.get_observation()
    gbest_val = pso.get_current_best_fitness()

    # Testing reset action
    preX = pso.X
    pso.reset_slow_particles()
    X = pso.X
    error = gbest_val - fDeltas[fun_num - 1]

    # Convert DataFrame
    df_velocity = pd.DataFrame(V)
    df_relative_fitness = pd.DataFrame(relative_fitness)
    df_replacement_batchcounts = pd.DataFrame(pbest_replacement_batchcounts)
    df_prepositions = pd.DataFrame(preX)
    df_positions = pd.DataFrame(X)

    # Save the DataFrame to a CSV file
    df_velocity.to_csv("observationsV.csv", index=False)
    df_relative_fitness.to_csv("observationsrelative_fitness.csv", index=False)
    df_replacement_batchcounts.to_csv("observationsreplacement_batchcounts.csv", index=False)
    df_prepositions.to_csv('positionsbefore.csv')
    df_positions.to_csv('positionsafter.csv')

    # Print the gbest_val and error
    print(f"Current Best Fitness (gbest_val): {gbest_val}")
    print(f"Error: {error}")
    print(f"Time: {run_time}")


runs = 1
dimensions = 30
swarm_size = 50
evals_per_dim = 10000
run_experiment(runs, dimensions, evals_per_dim, swarm_size)

