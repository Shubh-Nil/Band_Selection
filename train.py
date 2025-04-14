import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score

from utils import *

# individual = candidate = list of genes
# band = gene
# population = list of individuals
# pool - population (initially, or modified)

def genetic_algorithm():
    """
    Runs the Genetic Algorithm to select an optimal subset of spectral-bands.
        - Population initialization
        - Iteratively performs 'Crossover' and 'Mutation', to generate 'offspring'
        - Evaluated fitness of all candidates (both parents and offsprings)
        - Selects the top individuals for the next generation

    Returns -> the Best Candidate (set of spectral-band indices)
    """

    n = 10                                                                                               # Number of genes per individual's chromosome.
    ind = 8                                                                                              # Population size (Set a value, such that 'ind' and 'ind//2' are even)
    generations = 3                                                                                      # Number of 'generations' to run.
    num = 10                                                                                             # Number of positions in a population, to mutate
    
    # Load and preprocess the HyperSpectral Image.
    X, y = read_HSI()
    X_flat, y_flat = flatten_data(X, y)                                                                  # X_flat.shape = (C, H*W) = (200, 21025)
                                                                                                         # y_flat.shape = (H*W,) = (21025,)
    
    # Initialize the population.
    population = initiate_population(ind, n)
    print("Population initialized")
    
    # Run the genetic algorithm for a fixed number of generations.
    for gen in range(generations):
        print("\nGeneration:", gen)
        print("Current Population:\n", population)

        # 1.
        # Evaluate fitness for the entire population
        fitness_scores = fitness_all(population, X_flat, y_flat)
        print("Fitness scores:", fitness_scores)

        # Select the top 'ind//2' individuals based on fitness.
        top_indices = np.argsort(fitness_scores)[::-1][:ind//2]                                          # [:] = full list
                                                                                                         # [::-1] = full list in reverse (-1 means step backwards)
                                                                                                         # select the first 'ind//2' candidate indices, from the reversed sorted list
                                                                                                         # i.e. select the Top 'ind//2' candidate indices, with highest scores
        parents = population[top_indices]
        parent_fitness = fitness_scores[top_indices]
        print("Selected parents:\n", parents)
        
        # Convert parent fitness scores to selection probabilities
        fitness_sum = np.sum(parent_fitness)
        if fitness_sum == 0:
            parent_probs = np.full_like(parent_fitness, 1 / len(parent_fitness))                         # each parent having equal probabilities
        else:
            parent_probs = parent_fitness / fitness_sum
        
        offspring = []
        # Generate Mutated offspring-chromosomes, from parent-chromosomes

        # Standard selection
        # for i in range(0, len(parents) - 1, 2):
        #     parent1 = parents[i]
        #     parent2 = parents[i + 1]
        # 
        #     off1, off2 = crossover_single_point(parent1, parent2)
        #     offspring.append(mutate(off1))
        #     offspring.append(mutate(off2))


        # 2.
        # Roulette-Wheel selection  (Fitness proportional selection)
        while len(offspring) < (ind - len(parents)):
            # Sample two parents (with replacement)
            idx1, idx2 = np.random.choice(len(parents), size=2, replace=True, p=parent_probs)
            parent1 = parents[idx1]
            parent2 = parents[idx2]

            # 3. Crossover
            off1, off2 = crossover_uniform(parent1, parent2)
            offspring.append(off1)
            offspring.append(off2)
        
        # New population = parents + offspring (N/2 + N/2 = N)
        population = np.vstack((parents, np.array(offspring)))
        print("New population:\n", population)

        # 4. Mutation of the Population
        population = mutate_normal(population, num)                                                 # in python, the same 'population' will be modified 
        
    # Output the final results.
    print("\nFinal Population (Best Band Combinations):\n", population)
    print("Best individual:", population[0])                                                        # final candidate with highest score


if __name__ == '__main__':
    genetic_algorithm()