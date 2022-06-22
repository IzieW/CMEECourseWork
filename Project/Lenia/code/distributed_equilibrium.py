#!/usr/bin/env python3

"""Develop food source by distributing orbium equilibrium across environment.

FIRST ATTEMPT: Growth Mean"""

## IMPORTS ##
from lenia_package import *


## FUNCTIONS ##
def test_equilibrium_m(creature):
    """Test survival time as growth mean varies"""

    m = np.arange(-0.1, 0.17, 0.01)
    survival_time = list(np.zeros(len(m)))

    for i in range(len(m)):
        creature.m = m[i]
        creature.initiate()
        survival_time[i] = run_one(creature, enviro=None, verbose=False)

    return pd.DataFrame({
        "m": m,
        "survival_time": survival_time
    })

class Creature:
    """Defines a life form, their kernel and growth functions
    R: radius
    T: Time
    m: mean (growth function)
    s: standard deviation (growth function)
    b: kernel peaks (growth function)"""
    keys = ["R", "T", "m", "s", "nutrient", "b"]
    size = 64
    mid = size // 2

    bell = lambda x, m, s: np.exp(-((x - m) / s) ** 2 / 2)

    species_cells = {"orbium": [[0, 0, 0, 0, 0, 0, 0.1, 0.14, 0.1, 0, 0, 0.03, 0.03, 0, 0, 0.3, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0.08, 0.24, 0.3, 0.3, 0.18, 0.14, 0.15, 0.16, 0.15, 0.09, 0.2, 0, 0, 0,
                                 0],
                                [0, 0, 0, 0, 0, 0.15, 0.34, 0.44, 0.46, 0.38, 0.18, 0.14, 0.11, 0.13, 0.19, 0.18, 0.45,
                                 0, 0, 0],
                                [0, 0, 0, 0, 0.06, 0.13, 0.39, 0.5, 0.5, 0.37, 0.06, 0, 0, 0, 0.02, 0.16, 0.68, 0, 0,
                                 0],
                                [0, 0, 0, 0.11, 0.17, 0.17, 0.33, 0.4, 0.38, 0.28, 0.14, 0, 0, 0, 0, 0, 0.18, 0.42, 0,
                                 0],
                                [0, 0, 0.09, 0.18, 0.13, 0.06, 0.08, 0.26, 0.32, 0.32, 0.27, 0, 0, 0, 0, 0, 0, 0.82, 0,
                                 0],
                                [0.27, 0, 0.16, 0.12, 0, 0, 0, 0.25, 0.38, 0.44, 0.45, 0.34, 0, 0, 0, 0, 0, 0.22, 0.17,
                                 0],
                                [0, 0.07, 0.2, 0.02, 0, 0, 0, 0.31, 0.48, 0.57, 0.6, 0.57, 0, 0, 0, 0, 0, 0, 0.49, 0],
                                [0, 0.59, 0.19, 0, 0, 0, 0, 0.2, 0.57, 0.69, 0.76, 0.76, 0.49, 0, 0, 0, 0, 0, 0.36, 0],
                                [0, 0.58, 0.19, 0, 0, 0, 0, 0, 0.67, 0.83, 0.9, 0.92, 0.87, 0.12, 0, 0, 0, 0, 0.22,
                                 0.07],
                                [0, 0, 0.46, 0, 0, 0, 0, 0, 0.7, 0.93, 1, 1, 1, 0.61, 0, 0, 0, 0, 0.18, 0.11],
                                [0, 0, 0.82, 0, 0, 0, 0, 0, 0.47, 1, 1, 0.98, 1, 0.96, 0.27, 0, 0, 0, 0.19, 0.1],
                                [0, 0, 0.46, 0, 0, 0, 0, 0, 0.25, 1, 1, 0.84, 0.92, 0.97, 0.54, 0.14, 0.04, 0.1, 0.21,
                                 0.05],
                                [0, 0, 0, 0.4, 0, 0, 0, 0, 0.09, 0.8, 1, 0.82, 0.8, 0.85, 0.63, 0.31, 0.18, 0.19, 0.2,
                                 0.01],
                                [0, 0, 0, 0.36, 0.1, 0, 0, 0, 0.05, 0.54, 0.86, 0.79, 0.74, 0.72, 0.6, 0.39, 0.28, 0.24,
                                 0.13, 0],
                                [0, 0, 0, 0.01, 0.3, 0.07, 0, 0, 0.08, 0.36, 0.64, 0.7, 0.64, 0.6, 0.51, 0.39, 0.29,
                                 0.19, 0.04, 0],
                                [0, 0, 0, 0, 0.1, 0.24, 0.14, 0.1, 0.15, 0.29, 0.45, 0.53, 0.52, 0.46, 0.4, 0.31, 0.21,
                                 0.08, 0, 0],
                                [0, 0, 0, 0, 0, 0.08, 0.21, 0.21, 0.22, 0.29, 0.36, 0.39, 0.37, 0.33, 0.26, 0.18, 0.09,
                                 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0.03, 0.13, 0.19, 0.22, 0.24, 0.24, 0.23, 0.18, 0.13, 0.05, 0, 0, 0,
                                 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.06, 0.08, 0.09, 0.07, 0.05, 0.01, 0, 0, 0, 0, 0]]}

    def __init__(self, filename, dict=0, species="orbium", cluster=False, cx=20, cy=20, dir=0, n=1, injury_threshold = 10):
        """Initiate creature from parameters filename, or if file is false, load dictionary"""
        if filename:
            dict = {"organism_count": None, "nutrient": None}
            name = deepcopy(filename)
            # Load parameters #
            if cluster:
                filename = filename + "_parameters.csv"
            else:
                filename = "../parameters/" + filename.lower() + "_parameters.csv"
            with open(filename, "r") as f:
                csvread = csv.reader(f)
                for row in csvread:
                    if row[0] == "b":  # Where b is list of values
                        dict[row[0]] = [float(i) for i in row[1].strip("[]").split(",")]
                    else:
                        dict[row[0]] = float(row[1])
            self.name = name
        else:
            self.name = dict["orbium"]
        self.cells = Creature.species_cells[species]  # Cells only
        # Each parameter
        self.R = dict["R"]
        self.T = dict["T"]
        self.m = dict["m"]
        self.s = dict["s"]
        self.b = dict["b"][0]

        if not dict["nutrient"]:
            dict["nutrient"] = 0.001

        self.running_mean = deepcopy(dict["m"])

        self.nutrient = dict["nutrient"]

        self.cx = cx
        self.cy = cy

        self.injury = 0

        if dict["organism_count"]:
            self.n = int(dict["organism_count"])
        else:
            self.n = int(n)

        self.mutations = 0
        self.evolved_in = 0

        if dir:
            for i in range(dir):
                self.cells = np.rot90(self.cells)
        self.dir = dir

        self.survival_mean = dict["survival_mean"]  # Mean survival time in evolved environment
        self.survival_var = dict["survival_var"]  # Survival var in evolved environment

        self.A = 0
        self.K = self.kernel()
        self.enviro = 0  # Load and temporarily hold obstacle channels
        self.enviro_kernel = 0  # Load and temporarily hold obstacle kernels

        self.injury_threshold = injury_threshold

        self.initiate()  # load A

    def figure_world(A, cmap="viridis"):
        """Set up basic graphics of unpopulated, unsized world"""
        global img  # make final image global
        fig = plt.figure()  # Initiate figure
        img = plt.imshow(A, cmap=cmap, interpolation="nearest", vmin=0)  # Set image
        plt.title = ("World A")
        plt.close()
        return fig

    def figure_asset(self, cmap="viridis", K_sum=1, bar_K=False):
        """ Chart creature's kernel and growth based on parameter solutions
        Subplot 1: Graph of Kernel in matrix form
        Subplot 2: Cross section of Kernel around center. Y: gives values of cell in row, X: gives column number
        Subplot 3: Growth function according to values of U (Y: growth value, X: values in U)
        """
        R = self.R
        K = self.kernel(fourier=False)
        growth = self.growth
        K_size = K.shape[0];
        K_mid = K_size // 2  # Get size and middle of Kernel
        fig, ax = plt.subplots(1, 3, figsize=(14, 2),
                               gridspec_kw={"width_ratios": [1, 1, 2]})  # Initiate figures with subplots

        ax[0].imshow(K, cmap=cmap, interpolation="nearest", vmin=0)
        ax[0].title.set_text("Kernel_K")

        if bar_K:
            ax[1].bar(range(K_size), K[K_mid, :], width=1)  # make bar plot
        else:
            ax[1].plot(range(K_size), K[K_mid, :])  # otherwise, plot normally
        ax[1].title.set_text("K cross-section")
        ax[1].set_xlim([K_mid - R - 3, K_mid + R + 3])

        if K_sum <= 1:
            x = np.linspace(0, K_sum, 1000)
            ax[2].plot(x, growth(x))
        else:
            x = np.arange(K_sum + 1)
            ax[2].step(x, growth(x))
        ax[2].axhline(y=0, color="grey", linestyle="dotted")
        ax[2].title.set_text("Growth G")
        return fig

    def save(self, verbose=True, cluster=False):
        """Save creature configuration to csv"""
        if cluster:
            filename = self.name.lower() + "_parameters.csv"
        else:
            filename = "../parameters/" + self.name.lower() + "_parameters.csv"
        with open(filename, "w") as f:
            csvwrite = csv.writer(f)
            for i in Creature.keys:
                csvwrite.writerow([i, self.__dict__[i]])
            csvwrite.writerow(["mutations", self.mutations])
            csvwrite.writerow(["gradient", self.evolved_in])
            csvwrite.writerow(["survival_mean", self.survival_mean])
            csvwrite.writerow(["survival_var", self.survival_var])
            csvwrite.writerow(["organism_count", self.n])
        if verbose:
            print(self.name + " configuration saved to parameters/")

    def initiate(self, size=size, show=False):
        """Initiate learning channel with creature cell configurations"""
        A = np.zeros([size, size])
        new_A = deepcopy(A)
        if self.n > 1:  # if multiple orbium
            for i in range(self.n):
                temp = deepcopy(A)
                temp[np.random.randint(64), np.random.randint(64)] = 1  # seed grid randomely for each n of orbium
                c = self.cells
                for i in range(np.random.randint(4)):  # Rotate cells randomely
                    c = np.rot90(c)
                temp = convolve2d(temp, c, mode="same", boundary="wrap")  # populate grid with lenia
                new_A = sum([temp, new_A])
            self.A = new_A
        else:
            A[self.cx, self.cy] = 1
            A = convolve2d(A, self.cells, mode="same", boundary="wrap")  # Update grid
            self.A = A

        self.injury = 0
        self.running_mean = deepcopy(self.m)  # reset to starting mean value

    def kernel(self, mid=mid, fourier=True, show=False):
        """ Learning kernel for parameter solution. Default fourier transformed"""
        D = np.linalg.norm(np.ogrid[-mid:mid, -mid:mid]) / self.R  # define distance matrix
        """Take all cells within distance 1 and transform along gaussian gradient. 
        Produces a smooth ring-shaped kernel"""
        K = (D < 1) * Creature.bell(D, 0.5, 0.15)
        K = K / np.sum(K)  # normalise
        if show:
            plt.matshow(K)
        if fourier:
            K = np.fft.fft2(np.fft.fftshift(K))  # fourier transform kernel
        return K

    def growth(self, U):
        """Defines growth of cells based input neighbourhood sum, U
        and parameter configuration. Specifically, mean and standard deviation"""
        return Creature.bell(U, self.running_mean, self.s) * 2 - 1

    def obstacle_growth(self):
        """Obstacle growth function: Obstacle creates severe negative growth in Life form"""
        return -10 * np.maximum(0, (self.enviro - 0.001))

    def update(self, i):
        """Update creature according to any number of layered environmental grids"""
        global img
        U = np.real(np.fft.ifft2(self.K * np.fft.fft2(self.A)))  # Convolve by kernel to get neighbourhood sums
        self.A = np.clip(self.A + 1 / self.T * (self.growth(U) + sum([i.growth() for i in self.enviro])), 0, 1)
        img.set_array(sum([self.A, sum([i.grid for i in self.enviro])]))
        return img,

    def update_food(self, i):
        global img
        self.running_mean = self.running_mean - 0.001
        self.running_mean = self.running_mean + (overlap(self, self.enviro)*self.nutrient)

        U = np.real(np.fft.ifft2(self.K * np.fft.fft2(self.A)))  # Convolve by kernel to get neighbourhood sums
        self.A = np.clip(self.A + 1 / self.T * self.growth(U), 0, 1)
        img.set_array(sum([self.A, self.enviro.grid]))

    def update_killer_obstacle(self, i):
        global img
        U = np.real(np.fft.ifft2(self.K * np.fft.fft2(self.A)))  # Convolve by kernel to get neighbourhood sums
        self.A = np.clip(self.A + 1 / self.T * self.growth(U), 0, 1)
        self.enviro.growth(self)

        if self.injury > 10:
            self.A = np.zeros([self.size, self.size])

        img.set_array(sum([self.A, self.enviro.grid]))

    def update_naive(self, i):
        """Update learning channel by 1/T according to values in the learning channel"""
        global img
        U = np.real(np.fft.ifft2(self.K * np.fft.fft2(self.A)))  # Convolve by kernel to get neighbourhood sums
        self.A = np.clip(self.A + 1 / self.T * (self.growth(U)), 0, 1)  # Update A by growth function *1/T
        img.set_array(self.A)
        return img,

    def update_obstacle(self, i):
        """Update learning channel by 1/T according to values in the learning channel and obstacle channel"""
        global img
        U = np.real(np.fft.ifft2(self.K * np.fft.fft2(self.A)))  # Convolve by kernel to get neighbourhood sums
        self.A = np.clip(self.A + 1 / self.T * (self.growth(U) + self.obstacle_growth()), 0,
                         1)  # Update A by growth function *1/T
        img.set_array(sum([self.A, self.enviro]))
        return img,

    def update_obstacle_moving(self, i):
        """Update learning channel by 1/T according to values in the learning channel and obstacle channel"""
        global img
        U = np.real(np.fft.ifft2(self.K * np.fft.fft2(self.A)))  # Convolve by kernel to get neighbourhood sums
        self.enviro = convolve2d(self.enviro, self.enviro_kernel, mode="same", boundary="wrap")
        self.A = np.clip(self.A + 1 / self.T * (self.growth(U) + self.obstacle_growth()), 0,
                         1)  # Update A by growth function *1/T
        img.set_array(sum([self.A, self.enviro]))
        return img,

    def render(self, enviro, name=None):
        """Render orbium in any number of layered environments"""
        if name:
            name = "../results/" + name + "_anim.gif"
        else:
            name = "../results/" + self.name + "_anim.gif"
        print("Rendering animation...")
        self.initiate()
        self.enviro = enviro
        fig = Creature.figure_world(self.A + enviro.grid)
        anim = animation.FuncAnimation(fig, self.update_food, frames=200, interval=20)
        anim.save(name, writer="imagemagick")

    def render2(self, *enviro, name=None):
        """Render orbium in any number of layered environments"""
        if name:
            name = "../results/" + name + "_anim.gif"
        else:
            name = "../results/" + self.name + "_anim.gif"
        print("Rendering animation...")
        self.initiate()
        self.enviro = enviro
        fig = Creature.figure_world(self.A + sum([i.grid for i in enviro]))
        anim = animation.FuncAnimation(fig, self.update, frames=200, interval=20)
        anim.save(name, writer="imagemagick")

    def render_killer_ob(self, obstacle, name=None):
        """Render orbium in any number of layered environments"""
        if name:
            name = "../results/" + name + "_anim.gif"
        else:
            name = "../results/" + self.name + "_anim.gif"
        print("Rendering animation...")
        self.initiate()
        self.enviro = obstacle
        fig = Creature.figure_world(self.A + obstacle.grid)
        anim = animation.FuncAnimation(fig, self.update_killer_obstacle, frames=200, interval=20)
        anim.save(name, writer="imagemagick")

    def update_theta(self, muse):
        """Update parameters from parameters of input instance, muse.
        Update kernel"""
        self.R = muse.R
        self.T = muse.T
        self.s = muse.s
        self.m = muse.m
        self.b = muse.b
        self.K = muse.K

    def theta(self):
        for key in Creature.keys:
            print(key, self.__dict__[key])

    def show(self):
        plt.matshow(sum([self.A, self.enviro.grid]))


class Food:
    def __init__(self, n=3, r=8, nutrient=0.01):
        """Initiate food.
        r= radius of food bits
        n= number of food bits
        nutrient = nutrient given upon overlap with food"""
        self.n = n
        self.r = r
        self.nutrient = nutrient

        self.kernel = np.ones([r, r])
        self.grid = 0
        self.initiate()

    def initiate(self, size = Creature.size, seed=0):
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

def overlap(creature, feature):
    """Return sum of cells overlapping overlap"""
    overlapping = (sum([creature.A, feature.grid]) - 1) > 0
    return np.sum(overlapping*creature.A)


########## EVOLUTION  ############
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


def update_man(creature, food, moving=False, give_sums=False):
    """Update learning channel by 1/T according to values in learning channel A,
    and obstacle channel O"""
    creature.running_mean = creature.running_mean - 0.001  # minus 0.01 from growth mean for every time step
    creature.running_mean = creature.running_mean + (overlap(creature, food)*creature.nutrient)  # add nutrients from food

    U = np.real(np.fft.ifft2(creature.K * np.fft.fft2(creature.A)))
    creature.A = np.clip(creature.A + 1 / creature.T * creature.growth(U), 0, 1)

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
    x = np.random.randint(0, 5)
    mutant.__dict__[Creature.keys[x]] = mutate(mutant.__dict__[Creature.keys[x]])
    mutant.K = mutant.kernel()  # update mutant kernel

    # Run mutant and wild over runs number of obstacle configurations
    t_wild = np.zeros(runs)
    t_mutant = np.zeros(runs)
    for i in range(runs):
        enviro.initiate()  # configure environments
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
            obstacle.initiate(seed=i)  # set obstacle
            times[i - 1] = run_one(creature, obstacle, verbose=verbose)
    else:
        for i in range(1, runs + 1):
            creature.initiate()
            times[i - 1] = run_one(creature, obstacle, verbose=verbose)

    if summary:
        return times.mean(), times.var()
    else:
        return times


def show(creature, food):
    plt.matshow(sum([creature.A, food.grid]))
