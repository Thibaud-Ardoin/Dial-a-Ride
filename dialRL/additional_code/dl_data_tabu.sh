for VARIABLE in {1..10}
do
  curl http://neumann.hec.ca/chairedistributique/data/darp/tabu/pr0${VARIABLE}.res > /home/tibo/Documents/Prog/EPFL/own/data/instances/cordeau2003/res/res${VARIABLE}.txt
  # sed '1d' ./data/tabu${VARIABLE}.tmp >  ./data/tabu${VARIABLE}.csv
  # rm ./data/tabu${VARIABLE}.tmp
done

for VARIABLE in {10..20}
do
  curl http://neumann.hec.ca/chairedistributique/data/darp/tabu/pr${VARIABLE}.res > /home/tibo/Documents/Prog/EPFL/own/data/instances/cordeau2003/res/res${VARIABLE}.txt
  # sed '1d' ./data/tabu${VARIABLE}.tmp >  ./data/tabu${VARIABLE}.txt
  # rm ./data/tabu${VARIABLE}.tmp
done
