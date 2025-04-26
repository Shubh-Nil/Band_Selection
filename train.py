import numpy as np
# import scipy.io
# from sklearn.model_selection import train_test_split
# from sklearn import svm
# from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from utils import *

# np.random.seed(42)

# individual = candidate = list of genes
# band = gene
# population = list of individuals/ candidates
# pool - population (initially, or modified)

def genetic_algorithm(Image: str, NUM_BANDS: int, POP_SIZE: int, GENERATIONS: int) -> list:
    """
    Runs the Genetic Algorithm to select an optimal subset of spectral-bands.
        - Population initialization
        - Iteratively performs 'Crossover' and 'Mutation', to generate 'offspring'
        - Evaluated fitness of all candidates (both parents and offsprings)
        - Selects the top individuals for the next generation

    n = Subset number of bands (Number of genes per individual's chromosome)     
    POP_SIZE = Population size (Set a value, such that 'ind' and 'ind//2' are even)
    GENERATIONS = Number of 'generations' to run
        
    Returns -> the Best Candidate (set of spectral-band indices)
    """

    num = 10                                                                                             # Number of positions in a population, to mutate
    
    # Load and preprocess the HyperSpectral Image.
    X, y = read_HSI(Image = Image)
    X_flat, y_flat = flatten_data(X, y)                                                                  # X_flat.shape = (C, H*W) = (200, 21025)
                                                                                                         # y_flat.shape = (H*W,) = (21025,)
    band_pool, _ = X_flat.shape

    # Initialize the population.
    population = initiate_population(POP_SIZE, NUM_BANDS, band_pool)
    print("Population initialized")

    best_fitness_scores = []
    
    # Run the genetic algorithm for a fixed number of generations.
    for gen in range(GENERATIONS):
        print("\nGeneration:", gen)
        print("Current Population:\n", population)

        # 1.
        # Evaluate fitness for the entire population
        fitness_scores = np.array(fitness_all(population, X_flat, y_flat))
        print("Fitness scores:", fitness_scores)

        # Track best fitness for this generation
        best_fitness = fitness_scores.max()
        best_fitness_scores.append(best_fitness)

        # Select the top 'ind//2' individuals based on fitness.
        top_indices = np.argsort(fitness_scores)[::-1][:POP_SIZE//2]                                          # [:] = full list
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
        while len(offspring) < (POP_SIZE - len(parents)):
            # Sample two parents (with replacement)
            idx1, idx2 = np.random.choice(len(parents), size=2, replace=True, p=parent_probs)
            parent1 = parents[idx1]
            parent2 = parents[idx2]

            # 3. Crossover
            while True:
                off1, off2 = crossover_blending(parent1, parent2, band_pool)
                if len(set(off1)) == NUM_BANDS and len(set(off2)) == NUM_BANDS:
                    break


            offspring.append(off1)
            offspring.append(off2)
        
        # New population = parents + offspring (N/2 + N/2 = N)
        population = np.vstack((parents, np.array(offspring)))
        print("New population:\n", population)

        # 4. Mutation of the Population
        population = mutate_normal(population, num, band_pool)                                                 # in python, the same 'population' will be modified 
        
    # Output the final results.
    print("\nFinal Population (Best Band Combinations):\n", population)
    # print("Best individual:", population[0])                                                        # final candidate with highest score

    # # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(GENERATIONS), best_fitness_scores, marker='o')
    # plt.xlabel('Generation')
    # plt.ylabel('Best Fitness Score')
    # plt.title('Best Fitness Score vs Generation')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('fitness_vs_generation.jpg')
    # plt.show()
    # print("Saved fitness vs generation plot as 'fitness_vs_generation.jpg'.")

    return population[0]


if __name__ == '__main__':
    selected_bands = genetic_algorithm(
        Image = 'data/Indian_pines/Indian_pines_corrected.mat',
        NUM_BANDS = 10,
        POP_SIZE = 128,
        GENERATIONS = 150
    )
    print("Best individual:", sorted(selected_bands.tolist()))