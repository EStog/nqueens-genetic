import array
import random
from typing import MutableSequence, Optional, Sequence, Tuple

import deap.base
import deap.creator
import deap.tools
import numpy

from algorithms import eaSimpleWithElitism

# The definition of the Individual class must be set in module level in order multiprocessing to work.

# define a single objective, minimizing fitness strategy:
deap.creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))


# create the Individual class based on list of integers:
deap.creator.create("Individual",
                    array.array,
                    typecode='i',
                    fitness=deap.creator.FitnessMin)


# create the Individual class based on list of integers:
deap.creator.create("OptimizedIndividual",
                    array.array,
                    typecode='i',
                    fitness=deap.creator.FitnessMin)


def get_violations_count(individual: Sequence, queens_amount: int) -> Tuple[int]:
    """Get the amount of violations.
    A violation is counted if two queens are placed in the same diagonal.

    Args:
        individual (deap.creator.Individual): An individual
        queens_amount (int): The amount of queens

    Returns:
        int: the amount of violations
    """
    violations_amount = 0
    for i in range(queens_amount):
        for j in range(i+1, queens_amount):
            if abs(i-j) == abs(individual[i]-individual[j]):
                violations_amount += 1

    return violations_amount,


def variant_ag_optimized(
        queens_amount: int,
        population_size: int,
        cxpb: float,
        mutpb: float,
        ngen: int,
        toolbox: Optional[deap.base.Toolbox] = None,
    verbose: bool = __debug__
) -> Tuple[MutableSequence, Sequence, deap.tools.Logbook]:
    """This is the implementation of a variant solution to the N-Queens using Genetic Algorithm as proposed in [Wirsansky]_.

    The genotype is an array of integers, where repetition is not allowed. Each i-th value in the array specify the column of the i-th row where a queen is positioned. Tournament selection,  uniform partially matched crossover and shuffle mutation are used. The objective is to minimice the amount of violations (mutual-attacking queens) in the board.

    Adapted from [SolURL]_.

    .. [Wirsansky] Eyal Wirsansky, “Solving the N-Queens problem,” in Hands-On Genetic Algorithms with Python, Packt Publishing Ltd, 2020, pp. 128--136.

    .. [SolURL] https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python/blob/master/Chapter05/01-solve-n-queens.py

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
    toolbox.register("randomOrder", random.sample, range(queens_amount), queens_amount)

    # create the individual creation operator to fill up an Individual instance with shuffled indices:
    toolbox.register("individualCreator", deap.tools.initIterate,
                     deap.creator.Individual, toolbox.randomOrder)

    # create the population creation operator to generate a list of individuals:
    toolbox.register("populationCreator", deap.tools.initRepeat, list, toolbox.individualCreator)

    # fitness calculation - compute the total distance of the list of cities represented by indices:
    toolbox.register("evaluate", get_violations_count, queens_amount=queens_amount)

    # Genetic operators:
    toolbox.register("select", deap.tools.selTournament, tournsize=2)
    toolbox.register("mate", deap.tools.cxUniformPartialyMatched, indpb=2.0/queens_amount)
    toolbox.register("mutate", deap.tools.mutShuffleIndexes, indpb=1.0/queens_amount)

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=population_size)

    # prepare the statistics object:
    stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)

    # define the hall-of-fame object:
    hof = deap.tools.HallOfFame(30)

    population, logbook = eaSimpleWithElitism(
        population, toolbox, cxpb=cxpb, mutpb=mutpb,
        ngen=ngen, stats=stats, halloffame=hof, verbose=verbose)

    return population, hof.items[0], logbook
