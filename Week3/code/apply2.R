#!/usr/bin/Rscript --vanilla
# Author: Izie Wood
# Desc: Script illustrating apply using own function
# Date: Oct 2021

SomeOperation <- function(v) { # (What dos this function do?)
    if (sum(v) > 0){ #note that sum(v) is a single (scalar) value
        return(v * 100)
    }
    return (v)
}

M <- matrix(rnorm(100), 10, 10)
print(apply(M, 1, SomeOperation))