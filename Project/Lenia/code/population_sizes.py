# !/usr/bin/env python3

"""Script tests orbium evolution across different population sizes
to see if mesh is simply a local optima"""

from lenia_package import *

n = [10**2, 10**4, 10**6]
times = []
var = []
for i in n:
    orbium = Creature("orbium")
    obstacle = ObstacleChannel(n=5, r=8)
    optimise(orbium, obstacle, N=i)
    test = get_survival_time(orbium, obstacle, summary=True)
    times.append(test[0])
    var.append(test[1])

df = pd.DataFrame( {
    "population": n,
    "time": times,
    "variance": var
})

df.to_csv("../results/population_size_evolution")
