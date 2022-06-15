#!/bin/bash
# remember to add --save when you want to save the experiment log

SEED_ARRAY=(6 66 666 660 666 669)
F_ARRAY=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
P_ARRAY=(0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)


for i in {0..17}
do
  FUNC=${F_ARRAY[i]}
  for j in {0..8}
  do
    P=${P_ARRAY[j]}
    for k in {0..5}
    do
      SEED=${SEED_ARRAY[k]}
      option="--p ${P} --f ${FUNC} --seed ${SEED}"
      cmd="python GraphSAGE.py ${option}"
      eval $cmd
    done
  done
done
