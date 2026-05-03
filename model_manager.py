"""ModelManager for BAMM — mirrors FloodDiffusion's API so the Three.js UI
plugs in unchanged. Internally, BAMM is a batch generator: each text command
runs one full batch (~5–15 s on MPS), then frames are streamed out of the
FrameBuffer at the JS-side 20 FPS.

The Flask routes/JS expect: start_generation, update_text, pause_generation,
resume_generation, reset, get_next_frame, get_broadcast_frames,
get_buffer_status, plus `is_generating`, `frame_buffer`, `_base_schedule_config`,
`_base_cfg_config`, `model`. We expose all of these.
"""

import os
import sys
import threading
import time
from collections import deque
from os.path import join as pjoin
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


# --- BAMM imports ----------------------------------------------------------
# `gen_t2m` reads `vq_opt` and `opt` from its module globals (the original
# script ran the model loaders inside __main__ where those were locals). We
# inject them after constructing them locally — same trick used in the old
# ui_app.py.
import gen_t2m
from gen_t2m import (
    load_vq_model, load_trans_model, load_res_model, load_len_estimator,
)
from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.motion_process import recover_from_ric
from visualization.joints2bvh import Joint2BVHConvertor


# --- FrameBuffer (verbatim copy of FloodDiffusion's contract) --------------
class FrameBuffer:
    def __init__(self, target_buffer_size=4):
        self.buffer = deque(maxlen=400)
        self.target_size = target_buffer_size
        self.lock = threading.Lock()

    def add_frame(self, joints):
        with self.lock:
            self.buffer.append(joints)

    def get_frame(self):
        with self.lock:
            if len(self.buffer) > 0:
                return self.buffer.popleft()
            return None

    def size(self):
        with self.lock:
            return len(self.buffer)

    def clear(self):
        with self.lock:
            self.buffer.clear()


class _FakeModel:
    """The Flask /api/config endpoint expects `manager.model` to expose a few
    scalar attributes (chunk_size, noise_steps, cfg_scale). BAMM has no such
    object; we route to BAMM-relevant knobs instead."""
    pass


class ModelManager:
    def __init__(self):
        # ---- pin checkpoint identifiers from the README example ------------
        _fixed = [
            "--res_name", "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw",
            "--name", "2024-02-14-14-27-29_8_GPT_officialTrans_2iterPrdictEnd",
            "--gpu_id", "0",
            "--text_prompt", "placeholder",
            "--motion_length", "0",
            "--repeat_times", "1",
            "--ext", "ui_session",
        ]
        sys.argv = [sys.argv[0]] + _fixed
        opt = EvalT2MOptions().parse(is_eval=True)

        if opt.gpu_id == -1:
            opt.device = torch.device("cpu")
        elif torch.cuda.is_available():
            opt.device = torch.device("cuda:" + str(opt.gpu_id))
        elif torch.backends.mps.is_available():
            opt.device = torch.device("mps")
        else:
            opt.device = torch.device("cpu")
        print(f"[BAMM] device: {opt.device}")

        opt.nb_joints = 22
        self.opt = opt
        self.device = str(opt.device)

        # ---- load all BAMM components --------------------------------------
        root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
        model_opt = get_opt(pjoin(root_dir, "opt.txt"), device=opt.device)

        vq_opt_path = pjoin("./log/vq", opt.dataset_name, model_opt.vq_name, "opt.txt")
        vq_opt = get_opt(vq_opt_path, device=opt.device)
        vq_model, vq_opt = load_vq_model(vq_opt)
        model_opt.num_tokens = vq_opt.nb_code
        model_opt.num_quantizers = vq_opt.num_quantizers
        model_opt.code_dim = vq_opt.code_dim

        # gen_t2m.load_res_model / load_trans_model close over module globals
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

        self._vq_model = vq_model
        self._res_model = res_model
        self._t2m_transformer = t2m_transformer
        self._length_estimator = length_estimator

        self._mean = np.load(pjoin(
            "checkpoints", opt.dataset_name, model_opt.vq_name, "meta", "mean.npy"
        ))
        self._std = np.load(pjoin(
            "checkpoints", opt.dataset_name, model_opt.vq_name, "meta", "std.npy"
        ))

        # ---- state mirroring FloodDiffusion's ------------------------------
        self.frame_buffer = FrameBuffer(target_buffer_size=4)
        self.broadcast_frames = deque(maxlen=400)
        self.broadcast_id = 0
        self.broadcast_lock = threading.Lock()

        self.smoothing_alpha = 1.0   # BAMM frames are clean, no streaming smoothing needed
        self.history_length = 30     # not meaningful for batch; kept for UI parity

        self.current_text = ""
        self._last_generated_text = None
        self.is_generating = False
        self.should_stop = False
        self.generation_thread = None
        self._gen_lock = threading.Lock()
        self._seed = 1

        # BAMM-relevant knobs surfaced via the Config modal.
        # `motion_length` matches gen_t2m.py's --motion_length flag:
        #   0  → length predictor (sampled from softmax)
        #  -1  → max-length mode (predict end token)
        #   N  → fixed N frames (multiple of 4, max 196)
        self.model = _FakeModel()
        self.model.motion_length = 0
        self.model.time_steps = int(opt.time_steps)
        self.model.cond_scale = float(opt.cond_scale)
        self.model.temperature = float(opt.temperature)
        self._base_schedule_config = {
            "motion_length": 0,
            "time_steps": int(opt.time_steps),
        }
        self._base_cfg_config = {"cond_scale": float(opt.cond_scale)}

        # IK refinement (paper-faithful). Reuses gen_t2m.py's converter.
        # Loads the BVH template once at construction time.
        self._ik_converter = Joint2BVHConvertor()
        self.ik_enabled = True

        # Performance telemetry (always recorded; export via /api/export_performance).
        self._perf_lock = threading.Lock()
        self._perf = {
            "model": "BAMM",
            "device": str(opt.device),
            "session_started_at": time.time(),
            "batches": [],          # per-batch dicts (latency_ms, ik_ms, frames, prompt)
            "frames_pushed": 0,
            "frames_consumed": 0,
            "peak_buffer_size": 0,
            "ik_enabled": True,
        }

        print("[BAMM] ModelManager initialized successfully")

    # ---- batch generation ---------------------------------------------------
    def _generate_joints(self, prompt: str, motion_length: int = 0):
        """Run one full BAMM batch and return ((T, 22, 3) joints, ik_ms)."""
        fixseed(int(self._seed))
        captions = [prompt]

        if motion_length == 0:
            text_embedding = self._t2m_transformer.encode_text(captions)
            pred_dis = self._length_estimator(text_embedding)
            probs = F.softmax(pred_dis, dim=-1)
            token_lens = Categorical(probs).sample()
            is_predict_len = False
        elif motion_length == -1:
            token_lens = torch.LongTensor([196 // 4]).to(self.opt.device).long()
            is_predict_len = True
        else:
            token_lens = torch.LongTensor([int(motion_length) // 4]).to(self.opt.device).long()
            is_predict_len = False

        with torch.no_grad():
            mids, pred_len = self._t2m_transformer.generate(
                captions, token_lens,
                timesteps=int(self.model.time_steps),
                cond_scale=float(self.model.cond_scale),
                temperature=float(self.model.temperature),
                topk_filter_thres=self.opt.topkr,
                gsample=self.opt.gumbel_sample,
                is_predict_len=is_predict_len,
            )
            token_lens = pred_len
            m_length = int(token_lens[0]) * 4
            mids = self._res_model.generate(
                mids, captions, token_lens, temperature=1, cond_scale=5,
            )
            pred_motions = self._vq_model.forward_decoder(mids).detach().cpu().numpy()
            data = pred_motions * self._std + self._mean

        joint_data = data[0][:m_length]
        joints = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

        # IK refinement (matches BAMM's reference gen_t2m.py "_ik" output).
        # Foot-locking + 100-iter inverse kinematics removes ground slide.
        ik_ms = 0.0
        if self.ik_enabled:
            t0 = time.time()
            _, joints = self._ik_converter.convert(
                joints, filename=None, iterations=100, foot_ik=True,
            )
            ik_ms = (time.time() - t0) * 1000.0
        return joints, ik_ms  # (T, 22, 3), float

    # ---- worker -------------------------------------------------------------
    # Highest the buffer is allowed to grow before the worker pauses pushing.
    # 200 frames @ 20 FPS == 10 s of lookahead — plenty for smooth playback,
    # short enough that text changes show up quickly.
    _BUFFER_HIGH_WATER = 200

    def _generation_loop(self):
        print("[BAMM] generation loop started (continuous, no frame cap)")
        while not self.should_stop:
            text = self.current_text
            if not text:
                time.sleep(0.05); continue

            # Throttle: don't generate the next batch until the buffer is
            # below the high-water mark. JS drains at 20 FPS.
            if self.frame_buffer.size() >= self._BUFFER_HIGH_WATER:
                time.sleep(0.05); continue

            ml = int(self.model.motion_length)
            print(f"[BAMM] generating for: {text!r} (buf={self.frame_buffer.size()}, motion_length={ml})")
            t_batch = time.time()
            try:
                joints, ik_ms = self._generate_joints(text, motion_length=ml)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[BAMM] generation error: {e}")
                time.sleep(0.5)
                continue
            batch_ms = (time.time() - t_batch) * 1000.0

            # If text changed mid-gen, drop this batch's frames immediately.
            if self.should_stop or self.current_text != text:
                print("[BAMM] interrupted mid-batch by text change")
                continue

            with self._perf_lock:
                self._perf["batches"].append({
                    "prompt": text,
                    "frames": int(len(joints)),
                    "batch_ms": batch_ms,
                    "ik_ms": ik_ms,
                    "model_only_ms": batch_ms - ik_ms,
                    "ts": time.time(),
                })

            for frame in joints:
                if self.should_stop or self.current_text != text:
                    break
                # Wait for room in the buffer if the JS hasn't drained yet.
                while (self.frame_buffer.size() >= self._BUFFER_HIGH_WATER
                       and not self.should_stop
                       and self.current_text == text):
                    time.sleep(0.05)
                if self.should_stop or self.current_text != text:
                    break
                f = frame.copy()
                self.frame_buffer.add_frame(f)
                with self.broadcast_lock:
                    self.broadcast_id += 1
                    self.broadcast_frames.append((self.broadcast_id, f))
                with self._perf_lock:
                    self._perf["frames_pushed"] += 1
                    if self.frame_buffer.size() > self._perf["peak_buffer_size"]:
                        self._perf["peak_buffer_size"] = self.frame_buffer.size()
            print(f"[BAMM] pushed {len(joints)} frames; buf now {self.frame_buffer.size()}")
        print("[BAMM] generation loop stopped")

    # ---- public API mirroring FloodDiffusion's ModelManager -----------------
    def start_generation(self, text, history_length=None):
        with self._gen_lock:
            self.current_text = text
            self._last_generated_text = None  # force regen
            if history_length is not None:
                self.history_length = history_length
            if not self.is_generating:
                self.frame_buffer.clear()
                self.should_stop = False
                self.generation_thread = threading.Thread(
                    target=self._generation_loop, daemon=True,
                )
                self.generation_thread.start()
                self.is_generating = True

    def update_text(self, text):
        if text == self.current_text:
            return
        self.current_text = text
        self._last_generated_text = None
        self.frame_buffer.clear()

    def pause_generation(self):
        self.should_stop = True
        if self.generation_thread:
            self.generation_thread.join(timeout=2.0)
        self.is_generating = False

    def resume_generation(self):
        if self.is_generating:
            return
        self.should_stop = False
        self._last_generated_text = None
        self.generation_thread = threading.Thread(
            target=self._generation_loop, daemon=True,
        )
        self.generation_thread.start()
        self.is_generating = True

    def reset(self, history_length=None, smoothing_alpha=None):
        if self.is_generating:
            self.pause_generation()
        self.frame_buffer.clear()
        self.current_text = ""
        self._last_generated_text = None
        if history_length is not None:
            self.history_length = history_length
        if smoothing_alpha is not None:
            self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        # Restore knobs from base config
        self.model.motion_length = int(self._base_schedule_config["motion_length"])
        self.model.time_steps = int(self._base_schedule_config["time_steps"])
        self.model.cond_scale = float(self._base_cfg_config["cond_scale"])

    def get_next_frame(self):
        f = self.frame_buffer.get_frame()
        if f is not None:
            with self._perf_lock:
                self._perf["frames_consumed"] += 1
        return f

    def export_performance(self, out_dir: str = "."):
        """Snapshot the current perf stats to a JSON file. Returns the path."""
        import json
        import os
        with self._perf_lock:
            snap = dict(self._perf)
            snap["batches"] = list(snap["batches"])
        # Compute summary on the snapshot
        batches = snap["batches"]
        n = len(batches)
        if n:
            batch_ms = sorted(b["batch_ms"] for b in batches)
            ik_ms = sorted(b["ik_ms"] for b in batches)
            model_ms = sorted(b["model_only_ms"] for b in batches)
            frames = sum(b["frames"] for b in batches)
            session_s = time.time() - snap["session_started_at"]
            def pct(arr, p):
                if not arr: return 0.0
                k = max(0, min(len(arr) - 1, int(round((p / 100.0) * (len(arr) - 1)))))
                return arr[k]
            snap["summary"] = {
                "batches": n,
                "total_frames_generated": frames,
                "frames_pushed": snap["frames_pushed"],
                "frames_consumed": snap["frames_consumed"],
                "peak_buffer_size": snap["peak_buffer_size"],
                "session_seconds": session_s,
                "achieved_consume_fps": snap["frames_consumed"] / session_s if session_s > 0 else 0.0,
                "batch_ms_p50": pct(batch_ms, 50),
                "batch_ms_p95": pct(batch_ms, 95),
                "batch_ms_mean": sum(batch_ms) / n,
                "ik_ms_mean": sum(ik_ms) / n,
                "model_only_ms_mean": sum(model_ms) / n,
                "ik_overhead_pct": (sum(ik_ms) / sum(batch_ms) * 100.0) if sum(batch_ms) > 0 else 0.0,
            }
        else:
            snap["summary"] = {"batches": 0, "note": "no batches generated yet"}
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        path = os.path.join(out_dir, f"perf_BAMM_{ts}.json")
        with open(path, "w") as fp:
            json.dump(snap, fp, indent=2, default=str)
        return path

    def get_broadcast_frames(self, after_id, count=8):
        with self.broadcast_lock:
            frames = [
                (fid, joints) for fid, joints in self.broadcast_frames
                if fid > after_id
            ]
        return frames[:count]

    def get_buffer_status(self):
        return {
            "buffer_size": self.frame_buffer.size(),
            "target_size": self.frame_buffer.target_size,
            "is_generating": self.is_generating,
            "current_text": self.current_text,
            "smoothing_alpha": self.smoothing_alpha,
            "history_length": self.history_length,
            "schedule_config": dict(self._base_schedule_config),
            "cfg_config": dict(self._base_cfg_config),
        }


_model_manager = None


def get_model_manager():
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
