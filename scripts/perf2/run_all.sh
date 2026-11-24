source /mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/env.sh; cd $R
for t in run1 run2; do
  srun -p h100 --gres=gpu:1 -c 8 -t 00:30:00 bash scripts/perf2/run_bench4.sh $t e3nn,flash64,cueq,hybrid H2O water192 water648 water1536 > logs/perf2/$t.log 2>&1
done
