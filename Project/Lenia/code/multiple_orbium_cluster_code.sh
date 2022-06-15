# !/bin/bash
#PBS -lwalltime=7:00:00
#PBS -lselect=1:ncpus=1:mem=1gb
# Run simulations on the cluster, move results to home

module load anaconda3/personal

cp $HOME/lenia_package.py .
cp $HOME/orbium_parameters.csv .

echo "Script is about to run"

python3 $HOME/multiple_orbium.py

echo "Simulation complete, moving to Home..."

mv * $HOME

echo "Program has finished running"
