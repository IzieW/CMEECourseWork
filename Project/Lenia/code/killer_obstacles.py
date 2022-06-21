# !/usr/bin/env python3

"""Evolve lenia in enviro of killer obstacles: any cell overlap causes damage to entire field."""

from lenia_package import *


def overlap(creature, feature):
    """Define if overlap"""
    return ((sum([creature.A, feature.grid]) - 1) > 0).any()


class KillerObstacle:
    def __init__(self, n=3, r=8):
        """Initiate killer obstacles"""
        self.n = n
        self.r = r
        self.kernel = np.ones([r, r])
        self.grid = 0

        self.initiate()

    def initiate(self, size=Creature.size, seed=0):
        if seed:
            np.random.seed(seed)
        grid = np.zeros(size * size)
        grid[np.random.randint(0, len(grid), self.n)] = 1
        grid.shape = [size, size]
        # convolve by kernel shape
        self.grid = np.clip(convolve2d(grid, self.kernel, mode="same", boundary="wrap"), 0, 1)

    def show(self):
        plt.imshow(self.grid)
        plt.show()

    def growth(self, creature):
        overlap = (sum([creature.A, self.grid]) - 1) > 0
        creature.injury = sum([creature.injury, np.sum(creature.A * overlap)])

###### EVOLUTION #########
def mutate(p):
    """Mutate input parameter p"""
    return np.exp(np.log(p) + np.random.uniform(low=-0.2, high=0.2))


def prob_fixation(wild_time, mutant_time, N):
    """Return probability of fixation given time survived by mutant and wild type,
    and psuedo-population size N"""
    s = (mutant_time - wild_time) / wild_time  # selection coefficient

    """If s is zero, there is no selective difference between types and probability of 
    fixation is equal to 1/populdation size"""
    if s:
        return (1 - np.exp(-2 * s)) / (1 - np.exp(-2 * N * s))
    else:
        return 1 / N

def update_man(creature, enviro, moving=False, give_sums=False):
    """Update learning channel by 1/T according to values in learning channel A,
    and obstacle channel O"""
    U = np.real(np.fft.ifft2(creature.K * np.fft.fft2(creature.A)))
    creature.A = np.clip(creature.A + 1 / creature.T * creature.growth(U), 0, 1)
    for i in enviro: i.growth(creature)

    if creature.injury > creature.injury_threshold:  # Immediate kill
        creature.A = np.zeros([creature.size, creature.size])


def selection(t_wild, t_mutant):
    """Return winning solution based on survival times of wild type (t_wild) and
    mutant type (t_mutant)."""
    pfix = prob_fixation(wild_time=t_wild, mutant_time=t_mutant, N=population_size)  # get probability of fixation
    if pfix >= np.random.uniform(0, 1):
        # ACCEPT MUTATION
        return True
    else:
        # REJECT MUTATION
        return False


def run_one(creature, enviro, show_after=0, moving=False, verbose=True, give_sums=False):
    """Run creature of given parameters in given obstacle configuration until it dies.
    Show after specifies number of timesteps at when it will show what the grid looks like"""
    t = 0  # set timer
    global sums
    sums = np.zeros(10000)
    while np.sum(creature.A) and (
            t < 10000):  # While there are still cells in the learning channel, and timer is below cut off
        t += 1  # update timer by 1
        if verbose & (t % 1000 == 0):
            print(t)  # Show that it is working even after long waits
        if give_sums:
            sums[t - 1] = update_man(creature, enviro, moving=moving, give_sums=True)
        else:
            update_man(creature, enviro, moving=moving)  # Run update and show
        # if t == show_after:
        #   plt.matshow(sum([creature.A, obstacle.grid]))
    return t


def mutate_and_select(creature, enviro, moving=False, runs=100):
    """Mutate one parameter from creature and assess fitness of new solution agaisnt wild type
    in input obstacle environment. Save winning parameters to Creature.

    Method involve running wild type and mutant over ten distinct obstacle environment, and
    summing the survival time of each."""
    wild_type = creature
    mutant = deepcopy(creature)

    ## Choose parameter at random and mutate in mutant_type
    x = np.random.randint(0, 4)
    mutant.__dict__[Creature.keys[x]] = mutate(mutant.__dict__[Creature.keys[x]])
    mutant.K = mutant.kernel()  # update mutant kernel

    # Run mutant and wild over runs number of obstacle configurations
    t_wild = np.zeros(runs)
    t_mutant = np.zeros(runs)
    for i in range(runs):
        for e in enviro: e.initiate()  # configure environments
        wild_type.initiate()
        mutant.initiate()
        t_wild[i] = run_one(wild_type, enviro, moving=moving)
        t_mutant[i] = run_one(mutant, enviro, moving=moving)

    # Record mean and variance of survival times
    wild_mean = t_wild.mean()
    print(wild_mean)
    mutant_mean = t_mutant.mean()
    record_time(wild_mean=wild_mean,
                wild_var=t_wild.var(),
                mutant_mean=mutant_mean,
                mutant_var=t_mutant.var())

    # Select winning parameter
    if selection(wild_mean, mutant_mean):
        print("Accept mutation")
        creature.update_theta(mutant)  # Update creature parameters
        return True
    else:
        print("Reject mutation")
        return False


def optimise_timely(creature, obstacle, N, seed=0, run_time=10, moving=False, cluster=False, name=None):
    """Mutate and select input creature in psuedo-population of size N
    until wild type becomes fixed over fixation number of generations"""
    global population_size
    population_size = N
    np.random.seed(seed)  # set seed

    run_time = run_time * 60  # Translate to seconds
    """Evolve until parameters become fixed over fixation number of generations"""
    gen = 0  # time_count
    mutation = 0
    start = time.time()
    while (time.time() - start) < run_time:
        if mutate_and_select(creature, obstacle, moving=moving):  # Updates creature values
            mutation += 1
        gen += 1

    print("Saving configuration...")
    """Save winning parameters and timelogs"""

    if name:
        creature.name = name
    else:
        creature.name = str(creature.n) + "_orbium_t" + str(run_time) + "s" + str(seed) + "N" + str(
            N)

    """Update survival time mean and variance by running over 10 configurations with seed"""
    print("Calculating survival means...")
    survival_time = get_survival_time(creature, obstacle, summary=True)
    creature.mutations = mutation
    creature.survival_mean = survival_time[0]
    creature.survival_var = survival_time[1]
    if cluster:
        time_log.to_csv(creature.name + "_times.csv")  # Save timelog to csv
    else:
        time_log.to_csv("../results/" + creature.name + "_times.csv")  # Save timelog to csv
    creature.save(cluster=cluster)  # save parameters
    return 1


def get_survival_time(creature, obstacle=None, runs=10, summary=False, verbose=False):
    """Calculate average run time over seeded 10 configurations.
    Return mean and variance."""
    times = np.zeros(runs)
    if obstacle:
        for i in range(1, runs + 1):
            creature.initiate()  # Reset grid
            for o in obstacle: o.initiate(seed=i)  # set obstacle
            times[i - 1] = run_one(creature, obstacle, verbose=verbose)
    else:
        for i in range(1, runs + 1):
            creature.initiate()
            times[i - 1] = run_one(creature, obstacle, verbose=verbose)

    if summary:
        return times.mean(), times.var()
    else:
        return times
