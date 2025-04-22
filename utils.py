import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score

def read_HSI() -> tuple[np.ndarray, np.ndarray]:
    """
    Reads HyperSpectral Image (HSI) data and the Ground-truth labels from MATLAB .mat files.
    
    Returns:
        X : HSI with shape (H, W, C)
        y : Ground-truth labels with shape (H, W)
    """
    X = scipy.io.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
    y = scipy.io.loadmat('Indian_pines_gt.mat')['indian_pines_gt']
    return X, y

def flatten_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """    
    Each row corresponds to one spectral band (gene)  
    Each column of the X.flatten represents a pixel of X.
    
    Args:
        X : HSI with shape (H, W, C)
        y : Ground-truth labels with shape (H, W)
    
    Returns:
        X_flat : shape (C, H*W), where each row is a flattened band.
        y_flat : shape (H*W,)
    """
    H, W, C = X.shape
    X_flat = np.array([X[:, :, i].flatten() for i in range(C)])                         # Flatten each band (x[:,:,i]) into a 1D array, 
                                                                                        # and stack into a 2D array with shape (C, H*W)
    y_flat = y.flatten()
    return X_flat, y_flat

def initiate_population(ind: int, n: int) -> np.ndarray:
    """
    Initializes a Population for the genetic algorithm.
    
    Each individual is represented by a 'chromosome' containing n genes
    (selected spectral-band indices), chosen randomly from the possible set of genes.
    
    Args:
        n : Number of bands (genes) per individual.
        ind : Population size (Number of individuals).
        
    Returns: 
        population : array of shape (ind, n).
    """
    population = np.random.randint(0, 200, size=(ind, n))
    return population



def crossover_single_point(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs a single-point crossover between two parent chromosomes.
        - A random crossover point is chosen (excluding the endpoints) 
        - The genes are swapped between the two parents after that point
        - Create two offsprings.
    """
    # Choose a random crossover point (must be at least 1 and less than chromosome length)
    point = np.random.randint(1, len(parent1))
    print("Crossover point:", point)
    offspring1 = np.concatenate((parent1[:point], parent2[point:]))
    offspring2 = np.concatenate((parent2[:point], parent1[point:]))
    
    return offspring1, offspring2

def crossover_double_point(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs a double-point crossover between two parent chromosomes.
        - Two random crossover points are chosen (point1 < point2)
        - The segment between these points is swapped between the two parents
        - Returns two new offspring
    """

    point1 = np.random.randint(1, len(parent1) - 1)
    point2 = np.random.randint(point1 + 1, len(parent1))

    print(f"Double-point crossover between {point1} and {point2}")

    offspring1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
    offspring2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))

    return offspring1, offspring2

def crossover_uniform(parent1: np.ndarray, parent2: np.ndarray, p: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs uniform crossover between two parents.
        - For each gene, with crossover probability `p`, genes are swapped.
    """
    mask = np.random.rand(len(parent1)) < p                                             # swap-decision mask (boolean)
                                                                                        # 'len(parent1)' random numbers between 0 and 1
                                                                                        # if they are < p, then mask[i] is 'True'. Else 'False'.
    print("Uniform crossover mask:", mask)

    offspring1 = parent1.copy()
    offspring2 = parent2.copy()
    # Apply mask-based swapping (in-place)
    offspring1[mask], offspring2[mask] = parent2[mask], parent1[mask]                   # "swap" when mask[i] = True

    return offspring1, offspring2

def crossover_blending(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For each gene position i, the Offspring-genes are computed using a convex combination of its Parent-genes
    """
    # Sample beta_i 'Uniformly' from [0,1] for each gene
    beta = np.random.uniform(low=0.0, high=1.0, size=len(parent1))

    # Compute offspring genes
    offspring1 = np.clip(np.round(parent1 * beta + parent2 * (1 - beta)), 0, 199).astype(int)
    offspring2 = np.clip(np.round(parent2 * beta + parent1 * (1 - beta)), 0, 199).astype(int)

    return offspring1, offspring2



def mutate_random(population: np.ndarray, num: int) -> np.ndarray:
    """
    Mutates by randomly changing one gene to a new spectral band index
    """
    m, n = population.shape

    # Build a list of eligible positions 
    # 'Elitism' - Skip Candidate with the Highest Fitness Score
    eligible_positions = [(i, j) for i in range(1, m) for j in range(n)]

    # Choose 'num' positions randomly
    if num <= len(eligible_positions):
        chosen_indices = np.random.choice(len(eligible_positions), size=num, replace=False)             # without replacement
    else:
        chosen_indices = np.random.choice(len(eligible_positions), size=num, replace=True)              # with replacement

    # Apply Mutation: for each chosen position, assign a random band value in [0,199]
    for index in chosen_indices:
        i, j = eligible_positions[index]
        population[i, j] = np.random.randint(0, 200)

    return population

def mutate_normal(population: np.ndarray, num: int) -> np.ndarray:
    """
    For a given gene at position (i, j) with value p, the mutated value is computed as:
       p' = p + sigma * z
    """
    m, n = population.shape

    # Build a list of eligible positions 
    # 'Elitism' - Skip Candidate with the Highest Fitness Score
    eligible_positions = [(i, j) for i in range(1, m) for j in range(n)]

    # Choose 'num' positions randomly
    if num <= len(eligible_positions):
        chosen_indices = np.random.choice(len(eligible_positions), size=num, replace=False)             # without replacement
    else:
        chosen_indices = np.random.choice(len(eligible_positions), size=num, replace=True)              # with replacement

    # Apply Mutation: for each chosen position, assign a random band value in [0,199]
    for index in chosen_indices:
        i, j = eligible_positions[index]
        current_gene_value = population[i, j]

        # Standard deviation computed across all candidates (chromosomes) at gene position j
        sigma = np.std(population[:, j])
        # Sample z from standard normal distribution
        z = np.random.normal(0, 1)

        mutated_gene_value = current_gene_value + sigma * z
        mutated_value = int(np.clip(round(mutated_gene_value), 0, 199))
        population[i, j] = mutated_gene_value

    return population



def mean_class_vectors(candidate: np.ndarray, X_flat: np.ndarray, y_flat: np.ndarray) -> np.ndarray:
    """
    mean vector per class
    """
    
    # Extract only the selected bands from X_flat
    selected_X = X_flat[candidate, :]                                       

    # compute mean vector per class
    classes = np.unique(y_flat)
    means = []
    for c in classes:
        mask = (y_flat == c)
        if not np.any(mask):
            continue
        μ_c = selected_X[:, mask].mean(axis=1)  
        means.append(μ_c)
        
    means = np.stack(means, axis=0) 
    return means
    


def accuracy(candidate: np.ndarray, X_flat: np.ndarray, y_flat: np.ndarray, means: np.ndarray) -> float:
    """
    predicts class of each pixel vector using S.A.M principle
    """
    selected_X = X_flat[candidate, :]

    # compute dot‐products between each class-mean and each pixel
    dots = means @ selected_X

    # normalize to get cosθ → (K,1) * (1,N)
    mean_norms  = np.linalg.norm(means,     axis=1, keepdims=True)  # (K,1)
    pix_norms   = np.linalg.norm(selected_X, axis=0, keepdims=True)  # (1,N)
    cosines     = dots / (mean_norms * pix_norms + 1e-12)

    # pick the class with highest cosine sim
    preds = np.argmax(cosines, axis=0)

    return float((preds == y_flat).mean())

def average_distance(means: np.ndarray) -> float:
    """
    The mean over all pairwise distances between class‐mean vectors.
    """   

    # if there’s only one (or zero) class, no pairwise distance exists
    n = means.shape[0]
    if n < 2:
        return 0.0

    # sum up all pairwise distances
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total_dist += np.linalg.norm(means[i] - means[j])
            count += 1

    # average
    return total_dist / count

def average_correlation(candidate: np.ndarray, X_flat: np.ndarray) -> float:
    """
    Average Correlation = Average of Correlation values, for all the 'Unique' band pairs
    """
    # Extract only the selected bands from X_flat
    selected_X = X_flat[candidate, :] 
    
    # Compute the correlation matrix among the selected bands
    corr_matrix = np.corrcoef(selected_X)

    n = corr_matrix.shape[0]
    # Get indices of the upper triangle
    upper_tri_indices = np.triu_indices(n, k=1)                                                     # k=1 -> exclude the diagonal
                                                                                                    # k=0 -> include the diagonal
    # Calculate the average correlation from the 'unique' band pairs
    avg_corr = np.mean(corr_matrix[upper_tri_indices])
    
    return avg_corr

def fitness(candidate: np.ndarray, X_flat: np.ndarray, y_flat: np.ndarray) -> float:
    """
    fitness score = alpha(Accuracy) + beta(Avg. Euclidean Distance) + (1-alpha-beta)(Avg. Correlation)
    """
    alpha = 0.4
    beta = 0.3

    means = mean_class_vectors(candidate, X_flat, y_flat)

    acc = accuracy(candidate, X_flat, y_flat, means)
    avg_dist = average_distance(means)
    avg_corr = average_correlation(candidate, X_flat)

    fitness_score = alpha*acc + beta*avg_dist + (1-alpha-beta)*avg_corr

    print("Fitness:", fitness_score)
    return fitness_score

def fitness_all(pool: np.ndarray, X_flat: np.ndarray, y_flat: np.ndarray) -> list:
    """
    Evaluates the fitness scores for each candidate chromosome, in the entire population.
    """
    return [fitness(candidate, X_flat, y_flat) for candidate in pool]