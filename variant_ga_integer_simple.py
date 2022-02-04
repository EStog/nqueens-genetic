import array
import random
from typing import MutableSequence, Optional, Sequence, Tuple

import deap.algorithms
import deap.base
import deap.creator
import deap.tools
import numpy

# The definition of the Individual class must be set in module level in order multiprocessing to work.

# define a single objective, minimizing fitness strategy:
deap.creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))


# create the Individual class based on list of integers:
deap.creator.create("Individual",
                    array.array,
                    typecode='i',
                    fitness=deap.creator.FitnessMin)


def get_violations_count(individual: Sequence, queens_amount: int) -> Tuple[int]:
    """Get the amount of violations.
    A violation is counted if two queens are placed in the same column or are in the same diagonal.

    Args:
        individual (deap.creator.SimpleIndividual): An individual
        queens_amount (int): The amount of queens

    Returns:
        int: the amount of violations
    """
    non_violations_amount = 0
    for i in range(queens_amount):
        for j in range(i+1, queens_amount):
            if individual[i] == individual[j] or abs(i-j) == abs(individual[i]-individual[j]):
                non_violations_amount += 1

    return non_violations_amount,


def variant_ag_integer_simple(
        queens_amount: int,
        population_size: int,
        cxpb: float,
        mutpb: float,
        ngen: int,
        toolbox: Optional[deap.base.Toolbox] = None,
        verbose: bool = __debug__
) -> Tuple[MutableSequence, Sequence, deap.tools.Logbook]:
    """This is the implementation of a simple variant solution to the N-Queens using Genetic Algorithm.

    The genotype is an array of integers, where repetition is allowed. Each i-th value in the array specify the column of the i-th row where a queen is positioned. Roulette selection, one-point crossover and uniform integer mutation are used. The objective is to minimice the amount of violations (mutual-attacking queens) in the board.

    :param queens_amount: The size of the board and also the amount of queens. The board will always be considered an square matrix.
    :type queens_amount: int
    :param population_size: The size of the population
    :type population_size: int
    :param cxpb: Crossover probability
    :type cxpb: float
    :param mutpb: Mutation probability
    :type mutpb: float
    :param ngen: Amount of generations
    :type ngen: int
    :param toolbox: A toolbox with already defined functions. This is useful, for example, in case a different function ``map`` is needed.
    :type toolbox: Optional[deap.base.Toolbox]
    :param verbose: Whether to give extra console output or not
    :type verbose: bool
    :return: The final population, the best individual found and a class:`~deap.tools.Logbook` with the statistics of the evolution
    :rtype: Tuple[MutableSequence, Sequence, deap.tools.Logbook]
    """

    toolbox = toolbox or deap.base.Toolbox()

    # create an operator that generates randomly shuffled indices:
    toolbox.register("randomValues", random.choices, range(queens_amount), k=queens_amount)

    # create the individual creation operator to fill up an Individual instance with indices:
    toolbox.register("individualCreator", deap.tools.initIterate,
                     deap.creator.Individual, toolbox.randomValues)

    # create the population operator to generate a list of individuals:
    toolbox.register("populationCreator", deap.tools.initRepeat, list, toolbox.individualCreator)

    toolbox.register('evaluate', get_violations_count, queens_amount=queens_amount)
    toolbox.register("select", deap.tools.selTournament, tournsize=2)
    toolbox.register("mate", deap.tools.cxOnePoint)

    toolbox.register("mutate", deap.tools.mutUniformInt, low=0,
                     up=queens_amount-1, indpb=1.0/queens_amount)

    population = toolbox.populationCreator(n=population_size)

    # prepare the statistics object:
    stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)

    # define the hall-of-fame object:
    hof = deap.tools.HallOfFame(1)

    # perform the Genetic Algorithm flow:
    population, logbook = deap.algorithms.eaSimple(
        population, toolbox, cxpb=cxpb, mutpb=mutpb,
        ngen=ngen, stats=stats, halloffame=hof, verbose=verbose)

    return population, hof.items[0], logbook
