#!/bin/bash

rm contract_times.dat

for i in 4 6 8 12 16 20 24 32 40 48 64 80 96 112 128 192
do
  timeUS=`grep "cutensor contract ELAPSED" timingFiles/out${i}.dat | awk 'BEGIN {FS="="}; {print $2}' | awk '{print $1}'`
  memory=$(($i*$i*$i*$i*2*8))
  echo "$i $timeUS $memory" >> contract_times.dat

done
