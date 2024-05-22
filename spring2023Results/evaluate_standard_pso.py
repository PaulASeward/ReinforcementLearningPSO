from pso.pso_swarm import PSOSwarm
import pso.functions as functions
import os
import csv
from tqdm import tqdm

def train(self):
    results_dir = 'standard_pso_results'
    os.makedirs(results_dir, exist_ok=True)
    for function_number in tqdm(range(5, 29)):
        obj_f = functions.CEC_functions(dim=self.config.dim, fun_num=function_number)
        all_episodes_results = []

        for step in tqdm(range(1000)):
            self.swarm = PSOSwarm(objective_function=obj_f, config=self.config)
            self.swarm.reinitialize()
            episode_results = []
            for ep in range(self.config.num_episodes):
                self.swarm.optimize()
                best_fitness = self.swarm.get_current_best_fitness()
                episode_results.append(best_fitness)

            all_episodes_results.append(episode_results)

        all_episodes_results_np = np.array(all_episodes_results)

        # Calculate the average results per episode across all steps
        average_results_per_episode = np.mean(all_episodes_results_np, axis=0)

        # Define the file path for the current function number
        file_path = os.path.join(results_dir, f'f{function_number}.csv')

        # Write the average results to the CSV file
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Average Fitness'])
            for ep, avg_fitness in enumerate(average_results_per_episode):
                writer.writerow([ep, avg_fitness])