# !/usr/bin/env python3
"""Script to test ranges of orbium parameters"""

## Preparations ##
from lenia_package import *  # import package


## FUNCTIONS ##
def range_and_record(ranging=10):
    """Vary input parameter (p) by 0.1 and record rise/fall of
    survival time to find optimum"""
    o = Creature("orbium")
    parameters = [o.R]  # start parameter list
    zero_run = get_survival_time(o, obstacle=0, summary=True)
    times = [zero_run[0]]
    var = [zero_run[1]]
    for i in range(0, ranging):
        o.R = o.R + 0.2  # increment
        parameters.append(o.R)
        t = get_survival_time(o, obstacle=0, summary=True)
        times.append(t[0])
        var.append(t[1])

    df = pd.DataFrame(
        {
            "parameter": parameters,
            "time": times,
            "variance": var
        })

    return df
