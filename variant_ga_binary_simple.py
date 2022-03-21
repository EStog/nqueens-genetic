import array
import math
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
                    typecode='B',
                    fitness=deap.creator.FitnessMin)


def get_value(ind: Sequence, pos: int, queens_amount: int, bits_amount: int) -> int:
    """Gets the value in base 10 of the given position in the given individual

    :param ind: An individual
    :type ind: deap.creator.SimpleIndividual
    :param pos: The position of a row in the board
    :type pos: int
    :param queens_amount: The amount of queens in the board
    :type queens_amount: int
    :param bits_amount: The amount of bits that it is used to represent a column value
    :type bits_amount: int
    :return: The value of the column in the given row in base 10
    :rtype: int
    """
    v = 0

    # the position of the less significative bit of the value in pos.
    k = pos*bits_amount

    # the exponent of each summand in the representation
    e = 0

    while e < bits_amount and v < queens_amount:
        v += ind[k+e]*2**e
        e += 1

    # undo one step if the value is out of range
    if v >= queens_amount:
        v -= ind[k+e-1]*2**(e-1)

    return v


def get_violations_count(
        ind: Sequence,
        queens_amount: int,
        bits_amount: int
) -> Tuple[int]:
    """Get the amount of violations.

    A violation is counted if two queens are placed in the same column or are in the same diagonal.

    :param ind: An individual
    :type ind: deap.creator.SimpleIndividual
    :param queens_amount: The amount of queens
    :type queens_amount: int
    :param bits_amount: The amount of bits that it is used to represent a column value
    :type bits_amount: int
    :return: The amount of violations as a one element tuple
    :rtype: Tuple[int]
    """
    violations_amount = 0
    for i in range(queens_amount):
        for j in range(i+1, queens_amount):
            vi = get_value(ind, i, queens_amount, bits_amount)
            vj = get_value(ind, j, queens_amount, bits_amount)
            if vi == vj or abs(i-j) == abs(vi-vj):
                violations_amount += 1

    return violations_amount,


def variant_ag_binary_simple(
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

    # create an operator that randomly returns 0 or 1:
    toolbox.register("zeroOrOne", random.randint, 0, 1)

    bits_amount = math.ceil(math.log2(queens_amount))

    # create the individual operator to fill up an Individual instance:
    toolbox.register("individualCreator", deap.tools.initRepeat,
                     deap.creator.Individual, toolbox.zeroOrOne, bits_amount*queens_amount)

    # create the population operator to generate a list of individuals:
    toolbox.register("populationCreator", deap.tools.initRepeat, list, toolbox.individualCreator)

    toolbox.register('evaluate', get_violations_count,
                     queens_amount=queens_amount,
                     bits_amount=bits_amount)
    toolbox.register("select", deap.tools.selTournament, tournsize=2)
    toolbox.register("mate", deap.tools.cxOnePoint)
    toolbox.register("mutate", deap.tools.mutFlipBit, indpb=1.0/queens_amount)

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
