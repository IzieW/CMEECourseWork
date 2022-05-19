# !/usr/bin/bash
#PBS -lwalltime=07:00:00
#PBS -lselect=1:ncpus=1:mem=1gb
# Run simulations on the cluster, move results to home

module load anaconda3/personal 

cp $HOME/iw121_HPC_2021_main.R .

echo "Script is about to run"

env python3 < $HOME/evolving_orbium_cluster_code.py

echo "Simulation complete, moving to Home..."

mv orbium* $HOME

echo "Program has finished running"