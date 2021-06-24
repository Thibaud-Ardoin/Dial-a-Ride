echo "Generate a random data instance (data_file_generator.py)"
echo "stored at strategy/data/DARP_cordeau"

python dialRL/dataset/data_file_generator.py

echo "Run rf on that data (darp_restricted_fragments.py)"

python -O dialRL/strategies/external/darp_rf/darp_restricted_fragments.py 0

echo "Run the solution on the environment(complete_route.py). (Could not work in order of sol file)"

python dialRL/strategies/complete_route.py

