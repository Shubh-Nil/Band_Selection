import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from train import *

def frequency():
    band_counts = defaultdict(int)                                              # if you access a 'key' which is not initialized yet,
                                                                                # the 'value' will be automatically be initialized to 0    

    NUM_RUNS = 50
    NUM_BANDS = 10
    POP_SIZE = 100
    GENERATIONS = 150
    IMAGE_PATH = 'data/Indian_pines/Indian_pines_corrected.mat'

    for run in range(NUM_RUNS):
        print(f"Running GA iteration {run+1}/{NUM_RUNS}")
        selected_bands = genetic_algorithm(
            Image=IMAGE_PATH,
            NUM_BANDS=NUM_BANDS,
            POP_SIZE=POP_SIZE,
            GENERATIONS=GENERATIONS
        )
        for band in selected_bands:
            band_counts[band] += 1

    # Create frequency list for all bands
    total_bands = 200  # Indian Pines has 200 bands
    x = list(range(total_bands))
    y = [band_counts[i] for i in x]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(x, y)
    plt.xlabel("Spectral Band Index")
    plt.ylabel("Frequency")
    plt.title(f"Frequency of Band Selection over {NUM_RUNS} GA Runs")
    plt.grid(True)
    plt.savefig("Band_selection_frequency.jpg", format="jpg", dpi=300)
    plt.show()


if __name__ == '__main__':
    frequency()