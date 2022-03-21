import multiprocessing
import time

import deap.base
import deap.creator
import deap.tools
import pandas

from variant_ga_binary_simple import variant_ag_binary_simple
from variant_ga_integer_simple import variant_ag_integer_simple
from variant_ga_optimized import variant_ag_optimized

MIN_INDEX = 2
MAX_INDEX = 7

MIN_QUEENS_AMOUNT = 2**MIN_INDEX
MAX_QUEENS_AMOUNT = 2**MAX_INDEX

QUEENS_AMOUNTS = [2**i for i in range(MIN_INDEX, MAX_INDEX+1)]

POPULATION_SIZE = 300
GENERATION_NUMBER = 200

CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.1

RANDOM_SEED = 1500

VARIANTS = [variant_ag_binary_simple, variant_ag_integer_simple, variant_ag_optimized]
VARIANTS_NAMES = ['Simple Bin.', 'Simple Int.', 'Optimized']
VARIANTS_AMOUNT = len(VARIANTS)


def main():

    toolbox = deap.base.Toolbox()

    # prepare to parallel execution
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    bests = {name: [] for name in VARIANTS_NAMES}
    times = {name: [] for name in VARIANTS_NAMES}
    logbook = {name: None for name in VARIANTS_NAMES}

    for i in QUEENS_AMOUNTS:
        for variant, name in zip(VARIANTS, VARIANTS_NAMES):
            print(f'\nComputing {name} for {i} queens...')
            init = time.time()
            _, b, logbook[name] = variant(
                i, POPULATION_SIZE, CROSSOVER_PROBABILITY,
                MUTATION_PROBABILITY, GENERATION_NUMBER, toolbox,
                verbose=False)
            times[name].append(time.time() - init)
            bests[name].append(b.fitness.values[0])

    d = pandas.DataFrame(bests, index=QUEENS_AMOUNTS)
    d.to_csv('bests.csv')
    fig = d.plot(title='Bests fitness values per problem size',
                 xlabel='Queens amount',
                 ylabel='Fitness').get_figure()
    fig.savefig('bests.svg')

    d = pandas.DataFrame(times, index=QUEENS_AMOUNTS)
    d.to_csv('times.csv')
    fig = d.plot(title='Execution times per problem size',
                 xlabel='Queens amount',
                 ylabel='Execution time (s)').get_figure()
    fig.savefig('times.svg')

    mins_means = []
    graphs_names = []
    for name in VARIANTS_NAMES:
        mins_means.extend(logbook[name].select('min', 'avg'))
        graphs_names.extend((f'{name} min', f'{name} mean'))

    d = pandas.DataFrame(zip(*mins_means), columns=graphs_names)

    d.to_csv('stats.csv')
    fig = d.plot(title=f'Min/Mean fitness per generation for {MAX_QUEENS_AMOUNT} queens',
                 xlabel='Generation number',
                 ylabel='Fitness').get_figure()
    fig.savefig('stats.svg')


if __name__ == "__main__":
    main()
