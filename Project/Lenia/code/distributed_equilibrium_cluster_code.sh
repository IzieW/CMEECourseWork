# !/bin/bash
#PBS -lwalltime=13:00:00
#PBS -lselect=1:ncpus=1:mem=1gb
# Run simulations on the cluster, move results to home

module load anaconda3/personal

cp $HOME/distributed_equilibrium.py  .
cp $HOME/orbium_parameters.csv .

echo "Script is about to run"

python3 $HOME/distributed_equilibrium_cluster.py

echo "Simulation complete, moving to Home..."

mv * $HOME

echo "Program has finished running"
