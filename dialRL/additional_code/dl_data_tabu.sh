for VARIABLE in {1..10}
do
  curl http://neumann.hec.ca/chairedistributique/data/darp/tabu/pr0${VARIABLE} > ./data/tabu${VARIABLE}.tmp
  sed '1d' ./data/tabu${VARIABLE}.tmp >  ./data/tabu${VARIABLE}.csv
  rm ./data/tabu${VARIABLE}.tmp
done

for VARIABLE in {10..20}
do
  curl http://neumann.hec.ca/chairedistributique/data/darp/tabu/pr${VARIABLE} > ./data/tabu${VARIABLE}.tmp
  sed '1d' ./data/tabu${VARIABLE}.tmp >  ./data/tabu${VARIABLE}.csv
  rm ./data/tabu${VARIABLE}.tmp
done
