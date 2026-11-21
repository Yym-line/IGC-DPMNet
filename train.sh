#!/usr/bin/env bash 
set -eu  
epochs=120
# constrainted by GPU number & memory
batch_size=2
gpuid=0
num_workers=8
cpt_dir=/node/yym/expriment/IGC_DPMNet
#resume=
#[ $# -ne 1 ] && echo "Script error: $0 <gpuid> <cpt-id>" && exit 1
python ./nnet/train_unet_tse_steplr_clip.py \
  --gpu $gpuid \
  --epochs $epochs \
  --batch-size $batch_size \
  --num-workers $num_workers \
  --checkpoint $cpt_dir \
  #--resume  \
  #--lr 1e-4
> train.log 2>&1
