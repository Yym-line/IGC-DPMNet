#!/usr/bin/env python3

import os
import time
import argparse
import torch as th
import torch.nn.functional as F
import numpy as np
from IGC_DPMNet import FreqNet
from libs.utils import load_json, get_logger
from libs.audio import write_wav
from libs.dataset_tse import Dataset 

def chunk_inference(nnet, mix, aux, sr, chunk_sec=4.0, hop_sec=2.0):
    nnet.eval()
    chunk_size = int(chunk_sec * sr)
    hop_size = int(hop_sec * sr)
    T = mix.shape[-1]

    # 如果音频还没一个 chunk 长，直接补齐后推理再切回
    if T < chunk_size:
        pad = chunk_size - T
        mix_chunk = F.pad(mix, (0, pad), mode='constant')
        with th.no_grad():
            out = nnet(mix_chunk, aux)
        return out[..., :T]

    final = th.zeros(1, T).to(mix.device)
    weight = th.zeros(1, T).to(mix.device)

    start = 0
    while True:
        end = start + chunk_size
        
        # 边界处理：如果当前窗口超出了音频总长
        if end > T:
            # 强制让最后一次推理的窗口结束在音频的末尾
            start = T - chunk_size
            end = T
            is_last = True
        else:
            is_last = False

        # 提取切片
        mix_chunk = mix[:, start:end]
        
        with th.no_grad():
            # 推理
            out = nnet(mix_chunk, aux) # 假设输出维度为 (1, chunk_size)
            
        # 累加结果和权重（用于后续平均）
        final[:, start:end] += out
        weight[:, start:end] += 1.0

        if is_last:
            break
        start += hop_size

    # 对重叠部分取平均，消除拼接处的阶跃感
    final = final / (weight + 1e-8)
    return final

def load_average_model(nnet, checkpoint_files):
    """
    Load multiple checkpoints and average their model parameters.
    All checkpoints must have the same model architecture.
    """
    avg_state_dict = None

    for i, cpt_fname in enumerate(checkpoint_files):
        print(f"Loading checkpoint: {cpt_fname}")

        cpt = th.load(cpt_fname, map_location="cpu")
        state_dict = cpt["model_state_dict"]

        if avg_state_dict is None:
            avg_state_dict = {}

            for k, v in state_dict.items():
                if th.is_floating_point(v):
                    avg_state_dict[k] = v.clone()
                else:
                    # 非浮点参数/Buffer直接保留
                    avg_state_dict[k] = v.clone()

        else:
            for k, v in state_dict.items():
                if th.is_floating_point(v):
                    avg_state_dict[k] += v
                else:
                    # 非浮点buffer不做平均
                    avg_state_dict[k] = v.clone()

    num_models = len(checkpoint_files)

    for k in avg_state_dict:
        if th.is_floating_point(avg_state_dict[k]):
            avg_state_dict[k] /= num_models

    nnet.load_state_dict(avg_state_dict)

    print(f"Averaged {num_models} checkpoints.")

    return nnet
                                                                                                                                                
def run(args):
    start = time.time()
    logger = get_logger(
            os.path.join(args.checkpoint, 'separate.log'), file=True)
    dataset = Dataset(mix_scp=args.mix_scp, aux_scp=args.aux_scp, sample_rate=args.fs)

    # Load model
    nnet_conf = load_json(args.checkpoint, "mdl.json")
    nnet = FreqNet(**nnet_conf)

    checkpoint_files = [
        os.path.join(args.checkpoint, "best_ep37_loss-7.0811.pt.tar"),
        os.path.join(args.checkpoint, "best_ep35_loss-6.9584.pt.tar"),
        os.path.join(args.checkpoint, "best_ep26_loss-7.0824.pt.tar"),
    ]

    nnet = load_average_model(nnet, checkpoint_files)

    logger.info(
        "Average {} checkpoints for inference.".format(
            len(checkpoint_files)
        )
    )
    
    device = th.device(
        "cuda:{}".format(args.gpuid)) if args.gpuid >= 0 else th.device("cpu")
    nnet = nnet.to(device) if args.gpuid >= 0 else nnet
    nnet.eval()
    
    with th.no_grad():
        total_cnt = 0
        for i, data in enumerate(dataset):    
            mix = th.tensor(data['mix'], dtype=th.float32, device=device)
            aux = th.tensor(data['aux'], dtype=th.float32, device=device)
            key = data['key']
            if args.gpuid >= 0:
                mix = mix.unsqueeze(0).to(device)
                aux = aux.unsqueeze(0).to(device)
                
            # Forward            
            ests = chunk_inference(nnet, mix, aux, sr=args.fs,chunk_sec=4.0, hop_sec=2.0)
            ests = ests.squeeze(0).cpu().numpy()

            mix_np = mix.squeeze(0).cpu().numpy()
            norm = np.linalg.norm(mix_np, np.inf)

            ests = ests[:mix_np.shape[-1]]
            est_max = np.max(np.abs(ests))

            #energy_threshold=1e-4
            logger.info("Separate Utt{:d}".format(total_cnt + 1))
            ests = ests*norm/est_max
            #if est_max > energy_threshold:
            #    scale = (norm / est_max) * 0.9
            #    ests = ests * scale

            #else:
                # 如果声音太小，说明是静音段，我们保持原样或者直接降噪处理
                # 这样可以保证背景非常干净
            #    logger.info(f"Utt {key} detected as near-silent, skipping normalization to avoid noise.")
                # 可选：ests = ests * 0  # 强制静音

            # 5. 最终限幅保护（防止个别点溢出）
            #ests = np.clip(ests, -0.99, 0.99)
            
            fname = key
            write_wav(os.path.join(args.dump_dir, fname),
                      ests, fs=args.fs)
            total_cnt += 1   
    
    end = time.time()
    logger.info('Utt={:d} | Time Elapsed: {:.1f}s'.format(total_cnt, end-start))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Separating speech...')
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Directory of checkpoint")
    parser.add_argument("--gpuid", type=int, default=-1, 
                        help="GPU device to offload model to, -1 means running on CPU")
    parser.add_argument('--mix_scp', type=str, required=True,
                        help='mix scp')
    parser.add_argument('--ref_scp', type=str, required=False,
                        help='ref scp')
    parser.add_argument('--aux_scp', type=str, required=True,
                        help='aux scp')
    parser.add_argument('--fs', type=int, default=16000, 
                        help="Sample rate for mixture input")
    parser.add_argument('--dump-dir', type=str, default="/node/hzl/expriment/SEF_PNet_icassp2025_github/results",
                        help="Directory to dump separated results out")
    args = parser.parse_args()
    run(args)
