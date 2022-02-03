import multiprocessing

import deap.base
import matplotlib.pyplot as plt
import seaborn as sns

from variant_ga_optimized import variant_ag_optimized
from variant_ga_simple import variant_ag_simple

MIN_QUEENS_AMOUNT = 10
MAX_QUEENS_AMOUNT = 100
QUEENS_AMOUNT_STEP = 10
POPULATION_SIZE = 300
GENERATION_NUMBER = 100

CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.1

RANDOM_SEED = 1


def plot_bests(bests1, bests2):
    pass


def plot_last_stats(minFitnessValues1, meanFitnessValues1,
                    minFitnessValues2, meanFitnessValues2):
    plt.figure(1)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues1, color='darkred', label='Simple Variant Min')
    plt.plot(meanFitnessValues1, color='magenta', label='Simple Variant Mean')
    plt.plot(minFitnessValues2, color='darkblue', label='Optimized Variant Min')
    plt.plot(meanFitnessValues2, color='royalblue', label='Optimized Variant Mean')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')
    plt.legend()
    plt.savefig('stats.svg')

def main():

    toolbox = deap.base.Toolbox()

    # prepare to paralel execution
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    bests1 = []
    bests2 = []

    for i in range(MIN_QUEENS_AMOUNT, MAX_QUEENS_AMOUNT, QUEENS_AMOUNT_STEP):
        print(f'\nQueens Amount: {i}')
        _, best1, logbook1 = variant_ag_simple(
            i, POPULATION_SIZE, CROSSOVER_PROBABILITY,
            MUTATION_PROBABILITY, GENERATION_NUMBER, toolbox)
        _, best2, logbook2 = variant_ag_optimized(
            i, POPULATION_SIZE, CROSSOVER_PROBABILITY,
            MUTATION_PROBABILITY, GENERATION_NUMBER, toolbox)
        bests1.append(best1.fitness)
        bests2.append(best2.fitness)

    minFitnessValues1, meanFitnessValues1 = logbook1.select("min", "avg")
    minFitnessValues2, meanFitnessValues2 = logbook2.select("min", "avg")

    plot_bests(bests1, bests2)

    plot_last_stats(minFitnessValues1, meanFitnessValues1,
                    minFitnessValues2, meanFitnessValues2)


if __name__ == "__main__":
    main()
