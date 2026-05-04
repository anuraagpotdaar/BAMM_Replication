"""Standalone inference-delay benchmark for BAMM.

Loads the model directly (no Flask, no UI), runs `--warmup` warmup batches
followed by `--iterations` measured batches per seed (default seeds=1,2,3 →
30 measured iterations), writes a JSON snapshot of per-iteration timings,
peak memory, geometric quality metrics, and parameter counts to
`--output-dir`.

Mirrors the production live worker's pipeline exactly (length predictor →
M-Transformer → R-Transformer → VQ decode → recover_from_ric → IK), so the
numbers are comparable to perf_BAMM_*.json files emitted by the running app.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from os.path import join as pjoin

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def _collect_system_info(device: str) -> dict:
    vm = psutil.virtual_memory()
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_device": device,
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_matmul_precision": torch.get_float32_matmul_precision(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(vm.total / 1e9, 2),
        "ram_available_gb": round(vm.available / 1e9, 2),
    }
    if device.startswith("cuda"):
        try:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2,
            )
        except Exception:
            pass
    return info


def _check_resource_pressure(label: str) -> dict:
    cpu_pct = psutil.cpu_percent(interval=1.0)
    vm = psutil.virtual_memory()
    ram_used_pct = vm.percent
    warnings = []
    if cpu_pct >= 50:
        warnings.append(f"high CPU pressure: {cpu_pct:.0f}% busy across cores")
    if ram_used_pct >= 80:
        warnings.append(f"high RAM pressure: {ram_used_pct:.0f}% used "
                        f"({vm.available / 1e9:.1f} GB free)")
    if vm.available < 2 * 1024**3:
        warnings.append(f"low free RAM: only {vm.available / 1e9:.2f} GB available")
    snapshot = {
        "cpu_percent_1s": cpu_pct,
        "ram_used_percent": ram_used_pct,
        "ram_available_gb": round(vm.available / 1e9, 2),
        "warnings": warnings,
    }
    print(f"[{label}] resource snapshot: CPU {cpu_pct:.0f}%, RAM {ram_used_pct:.0f}% used")
    for w in warnings:
        print(f"[{label}] WARNING — {w}; numbers will be noisier than usual")
    return snapshot


def _param_count(*models) -> int:
    total = 0
    for m in models:
        if m is None:
            continue
        total += sum(p.numel() for p in m.parameters())
    return total


def _mem_snapshot(device_str: str) -> dict:
    info = {"rss_mb": round(psutil.Process().memory_info().rss / 1e6, 2)}
    if device_str.startswith("mps"):
        try:
            info["mps_allocated_mb"] = round(torch.mps.current_allocated_memory() / 1e6, 2)
            info["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 1e6, 2)
        except Exception:
            pass
    elif device_str.startswith("cuda"):
        try:
            info["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 2)
            info["cuda_max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 2)
        except Exception:
            pass
    return info


def _motion_quality_humanml3d(joints: np.ndarray) -> dict:
    """Geometric motion-quality proxies for the 22-joint HumanML3D skeleton.

    joints: (T, 22, 3) numpy. Returns:
      foot_skate_m_per_frame:  mean horizontal foot velocity while in ground
                               contact (low feet_y). Lower = cleaner contacts.
      min_joint_y_m:           lowest Y across all joints/frames. Negative =
                               ground penetration.
      root_xz_velocity_m_per_frame: mean root horizontal velocity. Sanity check
                               that motion exists.
    Foot indices (3, 4, 7, 8) match BAMM's `remove_fs(fid_l=(3,4), fid_r=(7,8))`.
    """
    if joints is None or len(joints) < 2:
        return {}
    foot_idx = [3, 4, 7, 8]
    feet = joints[:, foot_idx, :]
    feet_y = feet[..., 1]
    threshold_y = float(np.percentile(feet_y, 10)) + 0.05
    contact = feet_y <= threshold_y
    horiz_vel = np.linalg.norm(np.diff(feet[..., [0, 2]], axis=0), axis=-1)
    contact_pair = contact[:-1] & contact[1:]
    skate = float(horiz_vel[contact_pair].mean()) if contact_pair.any() else 0.0
    min_y = float(joints[..., 1].min())
    root_vel_xz = float(
        np.linalg.norm(np.diff(joints[:, 0, [0, 2]], axis=0), axis=-1).mean()
    )
    return {
        "foot_skate_m_per_frame": round(skate, 5),
        "min_joint_y_m": round(min_y, 4),
        "root_xz_velocity_m_per_frame": round(root_vel_xz, 5),
    }


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gen_t2m
from gen_t2m import (
    load_vq_model, load_trans_model, load_res_model, load_len_estimator,
)
from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.motion_process import recover_from_ric
from visualization.joints2bvh import Joint2BVHConvertor


def _pick_device(gpu_id: int) -> torch.device:
    if gpu_id == -1:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_models():
    fixed_argv = [
        "--res_name", "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw",
        "--name", "2024-02-14-14-27-29_8_GPT_officialTrans_2iterPrdictEnd",
        "--gpu_id", "0",
        "--text_prompt", "placeholder",
        "--motion_length", "0",
        "--repeat_times", "1",
        "--ext", "benchmark",
    ]
    saved_argv = list(sys.argv)
    sys.argv = [sys.argv[0]] + fixed_argv
    try:
        opt = EvalT2MOptions().parse(is_eval=True)
    finally:
        sys.argv = saved_argv

    opt.device = _pick_device(opt.gpu_id)
    opt.nb_joints = 22

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_opt = get_opt(pjoin(root_dir, "opt.txt"), device=opt.device)

    vq_opt_path = pjoin("./log/vq", opt.dataset_name, model_opt.vq_name, "opt.txt")
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_model, vq_opt = load_vq_model(vq_opt)
    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    gen_t2m.vq_opt = vq_opt
    gen_t2m.opt = opt

    res_opt_path = pjoin("checkpoints", opt.dataset_name, opt.res_name, "opt.txt")
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt)
    assert res_opt.vq_name == model_opt.vq_name
    t2m_transformer = load_trans_model(model_opt, opt, "latest.tar")
    length_estimator = load_len_estimator(model_opt)
    for m in (vq_model, res_model, t2m_transformer, length_estimator):
        m.eval().to(opt.device)

    mean = np.load(pjoin(
        "checkpoints", opt.dataset_name, model_opt.vq_name, "meta", "mean.npy"
    ))
    std = np.load(pjoin(
        "checkpoints", opt.dataset_name, model_opt.vq_name, "meta", "std.npy"
    ))
    converter = Joint2BVHConvertor()
    return opt, vq_model, res_model, t2m_transformer, length_estimator, mean, std, converter


def _run_one_batch(opt, vq_model, res_model, t2m_transformer, length_estimator,
                   mean, std, converter, prompt: str, seed: int):
    """Returns (total_ms, model_only_ms, ik_ms, frames, joints)."""
    fixseed(seed)
    captions = [prompt]

    t0 = time.time()
    text_embedding = t2m_transformer.encode_text(captions)
    pred_dis = length_estimator(text_embedding)
    probs = F.softmax(pred_dis, dim=-1)
    token_lens = Categorical(probs).sample()

    with torch.no_grad():
        mids, pred_len = t2m_transformer.generate(
            captions, token_lens,
            timesteps=opt.time_steps,
            cond_scale=opt.cond_scale,
            temperature=opt.temperature,
            topk_filter_thres=opt.topkr,
            gsample=opt.gumbel_sample,
            is_predict_len=False,
        )
        token_lens = pred_len
        m_length = int(token_lens[0]) * 4
        mids = res_model.generate(
            mids, captions, token_lens, temperature=1, cond_scale=5,
        )
        pred_motions = vq_model.forward_decoder(mids).detach().cpu().numpy()
        data = pred_motions * std + mean

    joint_data = data[0][:m_length]
    joints = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
    t_model = time.time()

    _, joints = converter.convert(joints, filename=None, iterations=100, foot_ik=True)
    t_ik = time.time()

    total_ms = (t_ik - t0) * 1000.0
    model_only_ms = (t_model - t0) * 1000.0
    ik_ms = (t_ik - t_model) * 1000.0
    return total_ms, model_only_ms, ik_ms, int(len(joints)), joints


def _aggregate_quality(qs: list) -> dict:
    if not qs:
        return {}
    keys = qs[0].keys()
    out = {}
    for k in keys:
        vals = [q[k] for q in qs if k in q]
        if vals:
            out[f"{k}_mean"] = round(float(np.mean(vals)), 5)
            out[f"{k}_std"] = round(float(np.std(vals, ddof=0)), 5)
    return out


def _summarize(name, device, prompt, seeds, warmup, measured, cold_start_ms,
               iters, peak_mem, param_counts, fps_playback,
               system=None, resources=None):
    warm_lat = sorted(it["latency_ms"] for it in iters)
    n = len(iters)

    def pct(arr, p):
        k = max(0, min(len(arr) - 1, int(round((p / 100.0) * (len(arr) - 1)))))
        return arr[k]

    mean_frames = sum(it["frames"] for it in iters) / max(n, 1)
    mean_total = sum(warm_lat) / max(n, 1)
    throughput = (mean_frames / mean_total) * 1000.0 if mean_total > 0 else 0.0
    sum_ik = sum(it.get("ik_ms", 0.0) for it in iters)
    sum_total = sum(warm_lat)
    ik_share = (sum_ik / sum_total) * 100.0 if sum_total > 0 else 0.0

    ms_per_frame = (mean_total / mean_frames) if mean_frames > 0 else 0.0
    ms_per_sec_motion = ms_per_frame * fps_playback

    per_seed = {}
    for it in iters:
        per_seed.setdefault(it["seed"], []).append(it["latency_ms"])
    seed_means = {s: round(statistics.mean(v), 2) for s, v in per_seed.items()}
    cross_seed_std = (
        round(statistics.stdev(seed_means.values()), 2)
        if len(seed_means) > 1 else 0.0
    )

    qualities = [it["quality"] for it in iters if it.get("quality")]
    quality_summary = _aggregate_quality(qualities)

    summary = {
        "cold_start_ms": cold_start_ms,
        "warm_mean_ms": mean_total,
        "warm_p50_ms": pct(warm_lat, 50),
        "warm_p95_ms": pct(warm_lat, 95),
        "warm_p99_ms": pct(warm_lat, 99),
        "warm_min_ms": min(warm_lat) if warm_lat else 0.0,
        "warm_max_ms": max(warm_lat) if warm_lat else 0.0,
        "warm_stddev_ms": statistics.stdev(warm_lat) if n > 1 else 0.0,
        "warm_seed_means_ms": seed_means,
        "warm_cross_seed_stddev_ms": cross_seed_std,
        "mean_frames_per_iteration": mean_frames,
        "throughput_fps_warm": throughput,
        "ms_per_frame": round(ms_per_frame, 3),
        "ms_per_second_of_motion_at_20fps": round(ms_per_sec_motion, 2),
        "ik_overhead_pct": ik_share,
        "peak_memory": peak_mem,
        "param_counts": param_counts,
        "quality": quality_summary,
    }
    return {
        "model": name,
        "device": device,
        "prompt": prompt,
        "seeds": seeds,
        "warmup_iterations": warmup,
        "measured_iterations": measured,
        "system": system or {},
        "resources_at_start": resources or {},
        "iterations": iters,
        "summary": summary,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _peak_merge(peak: dict, snap: dict):
    for k, v in snap.items():
        peak[k] = max(peak.get(k, 0.0), v)


def main():
    parser = argparse.ArgumentParser(description="BAMM inference benchmark (full)")
    parser.add_argument("--prompt", default="a person walks forward")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Measured iterations per seed (default: 10).")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seeds", default="1,2,3",
                        help="Comma-separated seeds. Total measured = "
                             "len(seeds) * iterations (default: 30).")
    parser.add_argument("--output-dir", default="/Users/anuraag/Developer/masters/ccn")
    parser.add_argument("--no-quality", action="store_true",
                        help="Skip geometric quality metrics.")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print(f"[BAMM-bench] loading model... seeds={seeds}, iters/seed={args.iterations}")
    opt, vq_model, res_model, t2m_transformer, length_estimator, mean, std, converter = _load_models()
    device = str(opt.device)
    print(f"[BAMM-bench] device={device}, prompt={args.prompt!r}")

    sys_info = _collect_system_info(device)
    print(f"[BAMM-bench] system: {sys_info['platform']}, "
          f"{sys_info['cpu_count_physical']}p/{sys_info['cpu_count_logical']}l cores, "
          f"{sys_info['ram_total_gb']:.1f} GB RAM, "
          f"torch={sys_info['torch_version']}")
    resources = _check_resource_pressure("BAMM-bench")

    param_counts = {
        "vq_model": _param_count(vq_model),
        "res_model": _param_count(res_model),
        "t2m_transformer": _param_count(t2m_transformer),
        "length_estimator": _param_count(length_estimator),
    }
    param_counts["total"] = sum(param_counts.values())
    print(f"[BAMM-bench] params: total={param_counts['total']:,} "
          f"(vq={param_counts['vq_model']:,}, res={param_counts['res_model']:,}, "
          f"t2m={param_counts['t2m_transformer']:,}, len={param_counts['length_estimator']:,})")

    peak_mem = {}
    cold_start_ms = None
    for w in range(max(args.warmup, 1)):
        seed = seeds[w % len(seeds)]
        total_ms, _, _, _, _ = _run_one_batch(
            opt, vq_model, res_model, t2m_transformer, length_estimator,
            mean, std, converter, args.prompt, seed,
        )
        if w == 0:
            cold_start_ms = total_ms
        _peak_merge(peak_mem, _mem_snapshot(device))
        print(f"[BAMM-bench] warmup {w} (seed={seed}): {total_ms:.1f} ms")

    iters = []
    for seed in seeds:
        for i in range(args.iterations):
            total_ms, model_ms, ik_ms, frames, joints = _run_one_batch(
                opt, vq_model, res_model, t2m_transformer, length_estimator,
                mean, std, converter, args.prompt, seed,
            )
            quality = {} if args.no_quality else _motion_quality_humanml3d(joints)
            mem = _mem_snapshot(device)
            _peak_merge(peak_mem, mem)
            iters.append({
                "id": len(iters),
                "seed": seed,
                "iter_idx": i,
                "latency_ms": total_ms,
                "model_only_ms": model_ms,
                "ik_ms": ik_ms,
                "frames": frames,
                "memory_after_mb": mem,
                "quality": quality,
            })
            print(f"[BAMM-bench] seed={seed} iter={i}: total={total_ms:.1f} ms "
                  f"(model={model_ms:.1f}, ik={ik_ms:.1f}), frames={frames}"
                  f"{', skate=' + format(quality.get('foot_skate_m_per_frame', 0), '.4f') if quality else ''}")

    snap = _summarize(
        "BAMM", device, args.prompt, seeds, args.warmup, len(iters),
        cold_start_ms, iters, peak_mem, param_counts,
        fps_playback=20, system=sys_info, resources=resources,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"bench_BAMM_{time.strftime('%Y%m%dT%H%M%S')}.json")
    with open(out, "w") as fp:
        json.dump(snap, fp, indent=2)
    print(f"[BAMM-bench] wrote {out}")
    s = snap["summary"]
    print(f"[BAMM-bench] warm mean: {s['warm_mean_ms']:.1f} ms,"
          f" p95: {s['warm_p95_ms']:.1f} ms,"
          f" cold: {s['cold_start_ms']:.1f} ms,"
          f" throughput: {s['throughput_fps_warm']:.1f} fps,"
          f" IK %: {s['ik_overhead_pct']:.1f}%,"
          f" ms/frame: {s['ms_per_frame']:.2f},"
          f" cross-seed σ: {s['warm_cross_seed_stddev_ms']:.1f} ms")
    if peak_mem:
        print(f"[BAMM-bench] peak memory: {peak_mem}")
    if s["quality"]:
        print(f"[BAMM-bench] quality: {s['quality']}")


if __name__ == "__main__":
    main()
