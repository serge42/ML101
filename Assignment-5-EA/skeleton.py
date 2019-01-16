#
# USI - Universita della svizzera italiana
#
# Machine Learning - Fall 2018
#
# Assignment 5: Evolutionary Algorithms
#
# (!) This code skeleton is just a recommendation, you don't have to use it.
#

import numpy as np
import matplotlib.pyplot as plt
import seaborn
import random as rand

from bitstring import BitArray


def get_fitness(chromosome):
    # Fitness = number of separated pairs of 1's
    fit = 0
    last = ''
    wait_zero = False
    for a in chromosome:
        if a == '0':
            last = a
            wait_zero = False
        elif last == '1' and not wait_zero:
            fit += 1
            wait_zero = True
        else:
            last = a
    return fit


def discrete_sample(probabilities):
    pass


def fitness_proportional_selection(fitnesses, population, length):
	total = sum(fitnesses)
	probabilities = [f/total for f in fitnesses]
	# print(probabilities)
	all_childs = []
	for i in range(int(length//2)):
		pA, pB = rand.choices(population, weights=probabilities, k=2)
		all_childs += two_point_crossover(pA, pB)
	# use parents for remaining population
	parents = len(population) - len(all_childs)
	all_childs += rand.choices(population, probabilities, k=parents)
	return all_childs


def bitflip_mutatation(chromosome, mutation_rate):
    if mutation_rate == 0:
        return chromosome
    r = rand.randrange(0, int(1/mutation_rate))
    if r == 0:  # do mutation
        point = rand.randrange(0, len(chromosome))
        chromosome = [a for a in chromosome]
        if chromosome[point] == '0':
            chromosome[point] = '1'
        else:
            chromosome[point] = '0'
        chromosome = ''.join(chromosome)

    return chromosome


def one_point_crossover(parentA, parentB):
    c_point = rand.randrange(1, len(parentA))
    childA = parentA[:c_point] + parentB[c_point:]
    childB = parentB[:c_point] + parentA[c_point:]
    return childA, childB


def two_point_crossover(parentA, parentB):
    sample = rand.sample(range(1, len(parentA)), 2)
    sample.sort()
    p1, p2 = sample
    childA = parentA[:p1] + parentB[p1:p2] + parentA[p2:]
    childB = parentB[:p1] + parentA[p1:p2] + parentB[p2:]
    return childA, childB


def generate_initial_population(length, population_size):
    # list of random integers whose last (length) bits range from 0..0 to 1..1
    population = [rand.randint(2**(length-1), 2**length)
                  for i in range(population_size)]
    # map to last (length) bits as strings (since bytes can only grow by 4 bits at a time)
    return list(map(lambda x: BitArray(hex=hex(x)).bin[-length:], population))


def ga(length, population_size, mutation_rate, cross_over_rate=0.1, max_gen=1000):
	"""
	length: length of chromosomes
	"""
	population = generate_initial_population(length, population_size)
	# print(population)
	run_stat = []
	for i in range(max_gen):
		fitnesses = [get_fitness(p) for p in population]
		all_childs = fitness_proportional_selection(
			fitnesses, population, cross_over_rate*len(population))
		population = list(
			map(lambda x: bitflip_mutatation(x, mutation_rate), all_childs))
		run_stat += [fitnesses]
		if i > 250:
			cross_over_rate = 0.2
			mutation_rate=0.04
		# if i > 1000:
		# 	cross_over_rate = 0.01
		# 	mutation_rate = 0.02

	# print(population)
	shit = 0
	return run_stat, shit


def plot_minmax_curve(run_stats):
    """
    Can't you fucking tell what this shit is attempting to achieve, fucktards!!!!
    """
    min_length = min(len(r) for r in run_stats)
    truncated_stats = np.array([r[:min_length] for r in run_stats])

    X = np.arange(truncated_stats.shape[1])
    means = truncated_stats.mean(axis=0)
    print(means)
    mins = truncated_stats.min(axis=0)
    maxs = truncated_stats.max(axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(means, '-o')

    ax.fill_between(X, mins[:, 0], maxs[:, 0], linewidth=0,
                    facecolor="b", alpha=0.3, interpolate=True)
    ax.fill_between(X, mins[:, 1], maxs[:, 1], linewidth=0,
                    facecolor="g", alpha=0.3, interpolate=True)
    ax.fill_between(X, mins[:, 2], maxs[:, 2], linewidth=0,
                    facecolor="r", alpha=0.3, interpolate=True)
    print('FUCKIGN PLOT')


def run_part1(length=50):
    run_stats = []
    for run in range(10):
        run_stat, _ = ga(length=length, population_size=50,
                         mutation_rate=0.01, max_gen=2000)
        run_stats.append(run_stat)
    plot_minmax_curve(run_stats)

# part 2


def rosenbrock(x):
    return np.sum((1-x[:-1])**2 + 100*(x[1:] - x[:-1]**2)**2, axis=0)


def plot_surface():
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    G = np.meshgrid(np.arange(-1.0, 1.5, 0.05), np.arange(-1.0, 1.5, 0.05))
    R = rosenbrock(np.array(G))

    fig = plt.figure(figsize=(14, 9))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(G[0], G[1], R.T, rstride=1,
                           cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.set_zlim(0.0, 500.0)
    ax.view_init(elev=50., azim=230)

    plt.show()


def sample_offspring(parents, lambda_, tau, tau_prime, epsilon0=0.001):
    pass


def ES(N=5, mu=2, lambda_=100, generations=100, epsilon0=0.001):
    pass


def plot_ES_curve(F):
    min_length = min(len(f) for f in F)
    F_plot = np.array([f[:min_length] for f in F])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.mean(F_plot.T, axis=1))
    ax.fill_between(range(min_length), np.min(F_plot.T, axis=1), np.max(
        F_plot.T, axis=1), linewidth=0, facecolor="b", alpha=0.3, interpolate=True)
    ax.set_yscale('log')


def run_part2(length=5):
    run_stats = []
    for i in range(10):
        fit, solution = ES(N=length, mu=10, lambda_=100,
                           epsilon0=0.0001, generations=500)
        run_stats.append(fit)
    plot_ES_curve(run_stats)


if __name__ == '__main__':
    # ga(length=5, population_size=10, mutation_rate=0.01, max_gen=30)
    run_part1()
    # mutated = bitflip_mutatation('00000000', 0)
    # print(mutated)
