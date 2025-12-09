# command

``` bash
export CUDA_VISIBLE_DEVICES=2
nohup python3 -u train_net.py \
     --num-gpus 1 \
     --config-file configs/radm.yaml \
     > ./log/dual_gram.log 2>&1 &

```