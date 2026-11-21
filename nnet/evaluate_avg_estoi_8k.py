import numpy as np

# 兼容旧版 mir_eval 调用 np.Inf
if not hasattr(np, "Inf"):
    np.Inf = np.inf

import os
import time
import argparse
import torch as th
import numpy as np
from mir_eval.separation import bss_eval_sources
from pesq import pesq as pesq2
from pypesq import pesq as pesq1
from pystoi.stoi import stoi
from IGC_DPMNet import FreqNet
from libs.utils import load_json, get_logger
from libs.dataset_tse import Dataset


# ==================== 新增函数1：平均多个模型参数 ====================
def average_models(model_files, checkpoint_dir, logger):
    """
    加载多个模型并平均它们的参数
    Args:
        model_files: list of str, 模型文件名列表，如 ["115.pt.tar", "118.pt.tar", "120.pt.tar"]
        checkpoint_dir: str, 模型文件所在目录
        logger: logger对象
    Returns:
        averaged_state_dict: dict, 平均后的参数字典
    """
    averaged_state_dict = {}

    for i, model_file in enumerate(model_files):
        cpt_fname = os.path.join(checkpoint_dir, model_file)
        cpt = th.load(cpt_fname, map_location="cpu")

        if i == 0:
            # 第一个模型，直接复制参数
            for key, value in cpt["model_state_dict"].items():
                averaged_state_dict[key] = value.clone().float()
        else:
            # 累加参数
            for key, value in cpt["model_state_dict"].items():
                averaged_state_dict[key] += value.float()

        logger.info(f"Loaded checkpoint from {cpt_fname}, epoch {cpt.get('epoch', 'unknown')}")

    # 计算平均值
    num_models = len(model_files)
    for key in averaged_state_dict.keys():
        averaged_state_dict[key] = averaged_state_dict[key] / num_models

    logger.info(f"Averaged {num_models} models")
    return averaged_state_dict


# ==================== 新增函数1结束 ====================

# ==================== 修改函数2：原evaluate函数改名为evaluate_single，并新增evaluate_averaged ====================
def evaluate_single(args, model_file, logger):
    """原evaluate函数，评估单个模型"""
    start = time.time()
    total_SISNR = 0
    total_SISNRi = 0
    total_PESQ = 0
    total_PESQi = 0
    total_PESQ2 = 0
    total_PESQi2 = 0
    total_STOI = 0
    total_STOIi = 0
    total_ESTOI = 0
    total_ESTOIi = 0
    total_SDR = 0
    total_cnt = 0

    # Load model
    nnet_conf = load_json(args.checkpoint, "mdl.json")
    nnet = FreqNet(**nnet_conf)
    cpt_fname = os.path.join(args.checkpoint, model_file)
    cpt = th.load(cpt_fname, map_location="cpu")
    nnet.load_state_dict(cpt["model_state_dict"])
    logger.info("Loaded checkpoint from {}, epoch {:d}".format(
        cpt_fname, cpt["epoch"]))

    device = th.device(
        "cuda:{}".format(args.gpuid)) if args.gpuid >= 0 else th.device("cpu")
    nnet = nnet.to(device) if args.gpuid >= 0 else nnet
    nnet.eval()

    # Load data
    if args.csv_path:
        dataset = Dataset(csv_path=args.csv_path, sample_rate=8000)
    else:
        dataset = Dataset(mix_scp=args.mix_scp, ref_scp=args.ref_scp, aux_scp=args.aux_scp, sample_rate=8000)

    with th.no_grad():
        for i, data in enumerate(dataset):
            mix = th.tensor(data['mix'], dtype=th.float32, device=device)
            aux = th.tensor(data['aux'], dtype=th.float32, device=device)

            if args.gpuid >= 0:
                mix = mix.unsqueeze(0).to(device)
                aux = aux.unsqueeze(0).to(device)

            # === 计算推理时间开始 ===
            infer_start = time.time()
            ests = nnet(mix, aux)
            infer_end = time.time()
            infer_time = infer_end - infer_start
            # === 计算推理时间结束 ===

            # Forward
            ref = data['ref']
            key = data['key']
            ests = ests.cpu().numpy().flatten()
            ref = data['ref'].flatten()
            mix = mix.squeeze(0).cpu().numpy().flatten()

            min_len = min(ests.size, ref.size, mix.size)
            ests = ests[:min_len]
            ref = ref[:min_len]
            mix = mix[:min_len]

            # Compute metrics
            if args.cal_sdr == 1:
                SDR, sir, sar, popt = bss_eval_sources(ref, ests)
                total_SDR += SDR[0]
            SISNR, delta = cal_SISNRi(ests, ref, mix)
            PESQ, PESQi, PESQ2, PESQi2 = cal_PESQi(ests, ref, mix)
            STOI, STOIi = cal_STOIi(ests, ref, mix)
            ESTOI, ESTOIi = cal_ESTOIi(ests, ref, mix)
            if args.cal_sdr == 1:
                logger.info(
                    "Utt={:d} | SDR={:.2f} | SI-SNR={:.2f} | SI-SNRi={:.2f} | PESQ={:.2f} | PESQi={:.2f}| PESQ2={:.2f} | PESQi2={:.2f} | | STOI={:.2f} | STOIi={:.2f} | ESTOI={:.3f} | ESTOIi={:.3f}".format(
                        total_cnt + 1, SDR[0], SISNR, delta, PESQ, PESQi, PESQ2, PESQi2, STOI, STOIi, ESTOI, ESTOIi))
            else:
                logger.info(
                    "Utt={:d} | SI-SNR={:.2f} | SI-SNRi={:.2f} | PESQ={:.2f} | PESQi={:.2f} | PESQ2={:.2f} | PESQi2={:.2f} | STOI={:.2f} | STOIi={:.2f} | ESTOI={:.3f} | ESTOIi={:.3f}".format(
                        total_cnt + 1, SISNR, delta, PESQ, PESQi, PESQ2, PESQi2, STOI, STOIi, ESTOI, ESTOIi))
            total_SISNR += SISNR
            total_SISNRi += delta
            total_PESQ += PESQ
            total_PESQi += PESQi
            total_PESQ2 += PESQ2
            total_PESQi2 += PESQi2
            total_STOI += STOI
            total_STOIi += STOIi
            total_ESTOI += ESTOI
            total_ESTOIi += ESTOIi
            total_cnt += 1
    end = time.time()

    logger.info('Time Elapsed: {:.1f}s'.format(end - start))
    if args.cal_sdr == 1:
        logger.info("Average SDR: {0:.2f}".format(total_SDR / total_cnt))
    logger.info("Average SI-SNR: {:.2f}".format(total_SISNR / total_cnt))
    logger.info("Average SI-SNRi: {:.2f}".format(total_SISNRi / total_cnt))
    logger.info("Average PESQ: {:.2f}".format(total_PESQ / total_cnt))
    logger.info("Average PESQi: {:.2f}".format(total_PESQi / total_cnt))
    logger.info("Average PESQ2: {:.2f}".format(total_PESQ2 / total_cnt))
    logger.info("Average PESQi2: {:.2f}".format(total_PESQi2 / total_cnt))
    logger.info("Average STOI: {:.2f}".format(total_STOI / total_cnt))
    logger.info("Average STOIi: {:.2f}".format(total_STOIi / total_cnt))
    logger.info("Average ESTOI: {:.3f}".format(total_ESTOI / total_cnt))
    logger.info("Average ESTOIi: {:.3f}".format(total_ESTOIi / total_cnt))


# ==================== 新增函数3：评估平均模型 ====================
def evaluate_averaged(args, model_files, logger):
    """评估多个模型平均后的参数"""
    start = time.time()
    total_SISNR = 0
    total_SISNRi = 0
    total_PESQ = 0
    total_PESQi = 0
    total_PESQ2 = 0
    total_PESQi2 = 0
    total_STOI = 0
    total_STOIi = 0
    total_ESTOI = 0
    total_ESTOIi = 0
    total_SDR = 0
    total_cnt = 0

    # 加载模型配置
    nnet_conf = load_json(args.checkpoint, "mdl.json")
    nnet = FreqNet(**nnet_conf)

    # ====== 修改点1：加载并平均多个模型 ======
    averaged_state_dict = average_models(model_files, args.checkpoint, logger)
    nnet.load_state_dict(averaged_state_dict)
    # ====== 修改点1结束 ======

    device = th.device(
        "cuda:{}".format(args.gpuid)) if args.gpuid >= 0 else th.device("cpu")
    nnet = nnet.to(device) if args.gpuid >= 0 else nnet
    nnet.eval()

    logger.info(f"Evaluating averaged model from {len(model_files)} checkpoints")

    # Load data (与evaluate_single相同)
    if args.csv_path:
        dataset = Dataset(csv_path=args.csv_path, sample_rate=8000)
    else:
        dataset = Dataset(mix_scp=args.mix_scp, ref_scp=args.ref_scp, aux_scp=args.aux_scp, sample_rate=8000)

    with th.no_grad():
        for i, data in enumerate(dataset):
            mix = th.tensor(data['mix'], dtype=th.float32, device=device)
            aux = th.tensor(data['aux'], dtype=th.float32, device=device)

            if args.gpuid >= 0:
                mix = mix.unsqueeze(0).to(device)
                aux = aux.unsqueeze(0).to(device)

            # === 计算推理时间开始 ===
            infer_start = time.time()
            ests = nnet(mix, aux)
            infer_end = time.time()
            infer_time = infer_end - infer_start
            # === 计算推理时间结束 ===

            # Forward
            ref = data['ref']
            key = data['key']
            ests = ests.cpu().numpy().flatten()
            ref = data['ref'].flatten()
            mix = mix.squeeze(0).cpu().numpy().flatten()

            min_len = min(ests.size, ref.size, mix.size)
            ests = ests[:min_len]
            ref = ref[:min_len]
            mix = mix[:min_len]

            # Compute metrics
            if args.cal_sdr == 1:
                SDR, sir, sar, popt = bss_eval_sources(ref, ests)
                total_SDR += SDR[0]
            SISNR, delta = cal_SISNRi(ests, ref, mix)
            PESQ, PESQi, PESQ2, PESQi2 = cal_PESQi(ests, ref, mix)
            STOI, STOIi = cal_STOIi(ests, ref, mix)
            ESTOI, ESTOIi = cal_ESTOIi(ests, ref, mix)
            if args.cal_sdr == 1:
                logger.info(
                    "Utt={:d} | SDR={:.2f} | SI-SNR={:.2f} | SI-SNRi={:.2f} | PESQ={:.2f} | PESQi={:.2f}| PESQ2={:.2f} | PESQi2={:.2f} | | STOI={:.2f} | STOIi={:.2f} | ESTOI={:.3f} | ESTOIi={:.3f}".format(
                        total_cnt + 1, SDR[0], SISNR, delta, PESQ, PESQi, PESQ2, PESQi2, STOI, STOIi, ESTOI, ESTOIi))
            else:
                logger.info(
                    "Utt={:d} | SI-SNR={:.2f} | SI-SNRi={:.2f} | PESQ={:.2f} | PESQi={:.2f} | PESQ2={:.2f} | PESQi2={:.2f} | STOI={:.2f} | STOIi={:.2f} | ESTOI={:.3f} | ESTOIi={:.3f}".format(
                        total_cnt + 1, SISNR, delta, PESQ, PESQi, PESQ2, PESQi2, STOI, STOIi, ESTOI, ESTOIi))
            total_SISNR += SISNR
            total_SISNRi += delta
            total_PESQ += PESQ
            total_PESQi += PESQi
            total_PESQ2 += PESQ2
            total_PESQi2 += PESQi2
            total_STOI += STOI
            total_STOIi += STOIi
            total_ESTOI += ESTOI
            total_ESTOIi += ESTOIi
            total_cnt += 1
    end = time.time()

    logger.info('Time Elapsed: {:.1f}s'.format(end - start))
    if args.cal_sdr == 1:
        logger.info("Average SDR: {0:.2f}".format(total_SDR / total_cnt))
    logger.info("Average SI-SNR: {:.2f}".format(total_SISNR / total_cnt))
    logger.info("Average SI-SNRi: {:.2f}".format(total_SISNRi / total_cnt))
    logger.info("Average PESQ: {:.2f}".format(total_PESQ / total_cnt))
    logger.info("Average PESQi: {:.2f}".format(total_PESQi / total_cnt))
    logger.info("Average PESQ2: {:.2f}".format(total_PESQ2 / total_cnt))
    logger.info("Average PESQi2: {:.2f}".format(total_PESQi2 / total_cnt))
    logger.info("Average STOI: {:.2f}".format(total_STOI / total_cnt))
    logger.info("Average STOIi: {:.2f}".format(total_STOIi / total_cnt))
    logger.info("Average ESTOI: {:.3f}".format(total_ESTOI / total_cnt))
    logger.info("Average ESTOIi: {:.3f}".format(total_ESTOIi / total_cnt))

def cal_SISNR(est, ref, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """ 
    assert len(est) == len(ref)
    est_zm = est - np.mean(est)
    ref_zm = ref - np.mean(ref)

    t = np.sum(est_zm * ref_zm) * ref_zm / (np.linalg.norm(ref_zm)**2 + eps)
        
    return 20 * np.log10(eps + np.linalg.norm(t) / (np.linalg.norm(est_zm - t) + eps))

def cal_SISNRi(est, ref, mix, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """ 
    assert len(est) == len(ref) == len(mix)
    sisnr1 = cal_SISNR(est, ref)
    sisnr2 = cal_SISNR(mix, ref)
    
    return sisnr1, sisnr1 - sisnr2
                         
def cal_PESQ(est, ref):
    assert len(est) == len(ref)
    mode ='nb'
    p = pesq1(ref, est,8000)
    p2 = pesq2(8000, ref, est, mode)
    return p,p2

def cal_PESQi(est, ref, mix):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(est) == len(ref) == len(mix)
    pesq1,pesq12 = cal_PESQ(est, ref)
    pesq2,pesq22= cal_PESQ(mix, ref)

    return pesq1, pesq1 - pesq2,pesq12,pesq12-pesq22

def cal_STOI(est, ref):
    assert len(est) == len(ref)
    p = stoi(ref, est, 8000)
    return p

def cal_STOIi(est, ref, mix):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(est) == len(ref) == len(mix)
    stoi1 = cal_STOI(est, ref)*100
    stoi2 = cal_STOI(mix, ref)*100

    return stoi1, stoi1 - stoi2


def cal_ESTOI(est, ref):
    """计算 ESTOI；pystoi 的 extended=True 启用扩展 STOI。"""
    assert len(est) == len(ref)
    return stoi(ref, est, 8000, extended=True)


def cal_ESTOIi(est, ref, mix):
    """返回 ESTOI 和 ESTOI improvement（相对混合语音的提升），均为 0-1 标度。"""
    assert len(est) == len(ref) == len(mix)
    estoi_est = cal_ESTOI(est, ref)
    estoi_mix = cal_ESTOI(mix, ref)
    return estoi_est, estoi_est - estoi_mix


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate separation performance using Conv-TasNet')
    parser.add_argument('--checkpoint', type=str,
                        default='/node/hzl/expriment/SEF_PNet_icassp2025_github/demo',
                        help='Path to model directory containing checkpoints')
    parser.add_argument('--gpuid', type=int, default=0,
                        help="GPU device to offload model to, -1 means running on CPU")
    # ── CSV 模式（新，推荐）──────────────────────────────────────────────────
    parser.add_argument('--csv_path', type=str, default=None,
                        help='CSV 文件路径（含 mix_path/src_path 列的 LibriSpeech 直接路径模式）；'
                             '指定此项时忽略 --mix_scp / --ref_scp / --aux_scp')
    # ── SCP 模式（旧，向后兼容）─────────────────────────────────────────────
    parser.add_argument('--mix_scp', type=str, default=None,
                        help='mix scp（VOiCES 模式，未指定 --csv_path 时使用）')
    parser.add_argument('--ref_scp', type=str, default=None,
                        help='ref scp（VOiCES 模式，未指定 --csv_path 时使用）')
    parser.add_argument('--aux_scp', type=str, default=None,
                        help='aux scp（VOiCES 模式，未指定 --csv_path 时使用）')
    parser.add_argument('--cal_sdr', type=int, default=None,
                        help='Whether calculate SDR, add this option because calculation of SDR is very slow')
    # ====== 修改点2：添加 --avg_models 参数 ======
    parser.add_argument('--avg_models', type=str, default=None,
                        help='要平均的模型文件名，用逗号分隔，如 "115.pt.tar,118.pt.tar,120.pt.tar"')
    # ====== 修改点2结束 ======

    args = parser.parse_args()

    # ====== 修改点3：根据参数选择评估方式 ======
    if args.avg_models is not None:
        # 平均模型模式
        model_files = [f.strip() for f in args.avg_models.split(',')]
        avg_log_file = os.path.join(args.checkpoint, "eval_averaged.log")
        avg_logger = get_logger(avg_log_file, file=True)
        avg_logger.info(f"Evaluating averaged model from: {model_files}")
        evaluate_averaged(args, model_files, avg_logger)
    else:
        # 原有模式：评估单个模型
        # eval best.pt.tar
        best_model_file = "best.pt.tar"
        best_log_file = os.path.join(args.checkpoint, "eval_best.log")
        best_logger = get_logger(best_log_file, file=True)
        best_logger.info(f"Evaluating model: {best_model_file}")
        evaluate_single(args, best_model_file, best_logger)

        # eval 110-122 epoch.pt.tar
        for epoch in range(110, 122):
            model_file = f"{epoch}.pt.tar"
            log_file = os.path.join(args.checkpoint, f"eval_{epoch}.log")
            logger = get_logger(log_file, file=True)
            logger.info(f"Evaluating model: {model_file}")
            evaluate_single(args, model_file, logger)
