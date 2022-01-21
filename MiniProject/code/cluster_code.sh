#!/bin/bash
#PBS -lwalltime=00:15:00
#PBS -lselect=1:ncpus=1:mem=1gb

module load anaconda3/personal 

echo "R is about to run"

R --vanilla < $HOME/iw121_HPC_2021_cluster.R

mv Q18_cluster_result* $HOME12

echo "R has finished running"