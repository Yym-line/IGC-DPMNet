#!/bin/bash 
set -eu

checkpoint=/node/yym/expriment/CIE-TF-Mamba/8k_024BLSTMmamba_decoder_mixboth
gpuid=0

data_root=/node/yym/node/CIE-mDPTNet_8/data/test

mix_scp=$data_root/mix_both.scp 
spk1_scp=$data_root/ref.scp 
aux_scp=$data_root/auxs1.scp 

cal_sdr=1



python ./nnet/evaluate_avg_estoi_8k.py \
  --checkpoint $checkpoint \
  --gpuid $gpuid \
  --mix_scp $mix_scp \
  --ref_scp $spk1_scp \
  --aux_scp $aux_scp \
  --cal_sdr $cal_sdr \
  --avg_models "best_ep67_loss-10.3548.pt.tar, best_ep68_loss-10.4981.pt.tar, best_ep69_loss-10.5587.pt.tar"
> eval.log 2>&1

echo "eval done!"
