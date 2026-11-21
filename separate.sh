#!/bin/bash 
set -eu
checkpoint=/node/yym/expriment/CIE-TF-Mamba/8k_024BLSTMmamba_decoderGS_WHAMR
gpuid=0

# CSV 文件路径（含 mix_path / src_path / t_enroll_start / t_enroll_end /
#               t_mixture_start / t_end 等列）
#csv_path=/node/yym/LibriVoices/4w_test.csv

data_root=/node/yym/WHAMR/scp/WHAMR/inturn/tt

mix_scp=$data_root/mix.scp 
aux_scp=$data_root/aux.scp 


fs=8000
dump_dir=/node/yym/expriment/CIE-TF-Mamba/8k_024BLSTMmamba_decoderGS_WHAMR/WHAMR

python ./nnet/separate_aver.py \
  --checkpoint $checkpoint \
  --gpuid $gpuid \
  --mix_scp $mix_scp \
  --aux_scp $aux_scp \
  --fs $fs \
  --dump-dir $dump_dir \
 > separate.log 2>&1

echo "Separate done!"
