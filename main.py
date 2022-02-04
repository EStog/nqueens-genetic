import multiprocessing

import deap.base
import deap.creator
import deap.tools
import matplotlib.pyplot as plt
import pandas

from variant_ga_optimized import variant_ag_optimized
from variant_ga_integer_simple import variant_ag_integer_simple

MIN_QUEENS_AMOUNT = 100
MAX_QUEENS_AMOUNT = 100
QUEENS_AMOUNT_STEP = 100
POPULATION_SIZE = 400
GENERATION_NUMBER = 200

CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.1

RANDOM_SEED = 1


def main():

    toolbox = deap.base.Toolbox()

    # prepare to parallel execution
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    bests1 = []
    bests2 = []

    for i in range(MIN_QUEENS_AMOUNT, MAX_QUEENS_AMOUNT+1, QUEENS_AMOUNT_STEP):
        print(f'\nComputing for {i} queens...')
        _, b1, logbook1 = variant_ag_integer_simple(
            i, POPULATION_SIZE, CROSSOVER_PROBABILITY,
            MUTATION_PROBABILITY, GENERATION_NUMBER, toolbox, verbose=False)
        _, b2, logbook2 = variant_ag_optimized(
            i, POPULATION_SIZE, CROSSOVER_PROBABILITY,
            MUTATION_PROBABILITY, GENERATION_NUMBER, toolbox, verbose=False)
        bests1.append(b1.fitness.values[0])
        bests2.append(b2.fitness.values[0])

    d = pandas.DataFrame(zip(bests1, bests2), index=range(
        MIN_QUEENS_AMOUNT, MAX_QUEENS_AMOUNT+1, QUEENS_AMOUNT_STEP), columns=['Simple', 'Optimized'])
    d.to_csv('bests.csv')
    d.plot(title='Bests fitness values per problem size',
           xlabel='Queens amount',
           ylabel='Fitness')
    plt.savefig('bests.svg')

    minFitnessValues1, meanFitnessValues1 = logbook1.select("min", "avg")
    minFitnessValues2, meanFitnessValues2 = logbook2.select("min", "avg")
    d = pandas.DataFrame(zip(minFitnessValues1, meanFitnessValues1, minFitnessValues2, meanFitnessValues2), columns=[
                         'Simple V. Min', 'Simple V. Mean', 'Optimized V. Min', 'Optimized V. Mean'])

    d.to_csv('stats.csv')
    d.plot(title=f'Min/Mean fitness per generation for {MAX_QUEENS_AMOUNT} queens',
           xlabel='Generation number',
           ylabel='Fitness')
    plt.savefig('stats.svg')


if __name__ == "__main__":
    main()
