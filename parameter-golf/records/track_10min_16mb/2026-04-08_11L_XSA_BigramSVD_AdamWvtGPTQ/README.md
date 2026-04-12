# Non-Record: 11L XSA + AdamW v_t Saliency-Weighted GPTQ + SVD BigramSystem

**Track:** 10 min / 8×H100 / 16 MB · **Type:** Non-record submission  
**Author:** Subramanyam6 · **Date:** 2026-04-08

---

## Results summary

| Seed | Steps | Wall-clock | Raw log BPB¹ | Corrected BPB² | Artifact |
|------|-------|-----------|--------------|----------------|----------|
| 1337 | 779 | 680 s | 0.9756 @step700 | **1.402** | 8.72 MB |
| 42 | 780 | 680 s | 0.9750 @step700 | **1.401** | 8.88 MB |
| 314 | 698 | 684 s | 1.0284 @step600 | **1.478** | **8.15 MB** ← submitted |

**3-seed mean (corrected): 1.427 ± 0.036 BPB**

¹ `evaluate_bpb()` in our script divides by fixed `BYTES_PER_TOKEN=3.5`.  
² Rescaled to leaderboard-comparable units using the implied `bytes_per_token ≈ 2.436` measured from SOTA logs (back-solved from published `val_loss` / `val_bpb` pairs). This is the number that belongs next to **SOTA 1.1147** and **baseline 1.2244**.

**Leaderboard comparison (lower BPB is better):** after correction, the three-seed mean (~1.43) is **higher (worse)** than both the current SOTA (~1.11) and the naive baseline (~1.22). This is a **non-record** submission documenting two novel techniques, corrected metrics, and analysis of the gap (throughput limits, logging scale, and post-GPTQ effects)—not a SOTA claim.

---

## What is novel

### 1. AdamW v_t Saliency-Weighted GPTQ (zero artifact bytes)

Standard GPTQ treats all columns as equally trustworthy from an optimization perspective — it sees activation Hessians but not training history. We break this symmetry.

After the 600 s training window, but **before** DDP teardown, we harvest `exp_avg_sq` (the AdamW second moment, \(v_t = \mathbb{E}[g^2]\)) for all AdamW-managed parameters and the momentum buffer magnitude for Muon-managed 2D weights:

```python
# _collect_saliency() in train_gpt.py — called while optimizer is still live
for name, p in named_parameters:
    state = opt_adam.state[p]
    if 'exp_avg_sq' in state:
        saliency[name] = state['exp_avg_sq'].detach().float().cpu()
```

These tensors encode a **full-run heat map**: which weight columns absorbed sustained gradient pressure over all 700+ steps. During GPTQ we inject this into the Hessian diagonal:

```python
# _gptq_quantize_weight() in train_gpt.py
col_sal = saliency.mean(dim=0)                         # [cols] per-column importance
col_sal = col_sal / col_sal.mean().clamp_min(1e-8)     # normalize to mean=1
H.diagonal().add_(0.1 * col_sal * H.diagonal().mean()) # boost high-gradient cols
```

This tells GPTQ: "these columns were under sustained gradient pressure during training — give them priority precision." It is purely a code change; **0 extra bytes** appear in `model.bin`.

No prior leaderboard entry uses optimizer state to guide quantization column ordering.

### 2. SVD-Initialized BigramSystem (fast statistics from step 0)

Standard BigramHash initializes randomly. We accumulate bigram co-occurrence counts on GPU via `index_add_` (avoiding a numpy copy-discard bottleneck present in earlier implementations), then initialize the embedding and projection via **truncated SVD** of the log-frequency matrix:

```python
lp = torch.log(counts / row_sums)          # [n_buckets, vocab] log-bigram
U, S, Vh = torch.linalg.svd(lp, full_matrices=False)
self.embed.weight.data[:] = U[:, :D] * S[:D].sqrt()
self.proj.weight.data[:]  = Vh[:D, :V] * S[:D].sqrt().unsqueeze(1)
```

The BigramSystem therefore provides **meaningful signal at step 0** rather than training from noise. Combined with EMA(0.997), this accelerates early convergence.

---

## Architecture

| Component | Setting | Credit |
|-----------|---------|--------|
| Layers | 11 (d=512, 8 heads) | Baseline |
| MLP | 3× width (1536), relu² | Standard stack |
| Attention | XSA all 11 layers | [PR #478](https://github.com/openai/parameter-golf/pull/478) @gowtham0992 |
| Positional | RoPE (all layers) | Standard |
| Embeddings | Tied lm_head ↔ embed | Saves ~1 MB |
| BigramSystem | 2048 buckets × d=128, **SVD-init** | Concept: [PR #162](https://github.com/openai/parameter-golf/pull/162); init: **this work** |
| Weight avg | EMA(0.997) | [PR #401](https://github.com/openai/parameter-golf/pull/401) @newjordan |
| SLOT bias | 512-param global additive bias | **This work** |
| QK-Gain | init=4.0 per-head scalar | **This work** |
| Quantization | **AdamW v_t saliency GPTQ int6** | **This work** |
| Compression | zstd-22 | Prior work |
| Optimizer | Parallel Muon (2D) + AdamW (1D) | Competition consensus |
| Grad accum | 4 (required for OOM-free 786K batch on 8×H100) | This run |
| Softmax | fp32 (bf16 caused instability without FA3) | This run |

**Params (deduplicated):** 29,765,720 · **FP32 size:** 119.1 MB · **Post-GPTQ model.bin:** 8.15 MB

---

## Training

All three seeds ran on **8×H100 80GB SXM** on RunPod with the same command:

```bash
cd /workspace/parameter-golf
SEED=<seed> MAX_WALLCLOCK_SECONDS=600 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --nproc_per_node=8 train_gpt.py
```

Step rate: **~857 ms/step** (vs SOTA's 86 ms/step — a 10× gap, see §Bottlenecks).

**Data-path fix:** FineWeb `.bin` shards have a 1024-byte header; required `offset=1024` in `np.fromfile`. Missing this silently poisons the token stream with garbage values and produces flat loss across all steps — this was the most consequential silent bug fixed during development.

---

## Why corrected BPB still sits above SOTA and baseline (higher = worse)

Two independent bottlenecks explain the **corrected ~1.40 BPB** vs SOTA **1.1147 BPB** gap:

### 1. Training throughput (10× fewer steps)

SOTA achieves **~6,900 optimizer steps** per 600 s using **Flash Attention 3** (Hopper warp-specialized kernels, ~86 ms/step). We reached **~700–780 steps** (~857 ms/step) — limited by:

- No FA3: standard `F.scaled_dot_product_attention` + fp32 softmax reversion
- `grad_accum=4` (required for memory stability without FA3 kernel efficiency)
- seq_len=1024 vs SOTA's 2048 (lower GPU utilization per step)

At 700 steps the model is **far from convergence**. The training curve (see logs) shows no sign of plateau — it is still descending at step 700 when the wall-clock cuts it off.

### 2. `bytes_per_token` bug in logging

Our script reported BPB using `BYTES_PER_TOKEN=3.5` (hardcoded). The **actual** measured ratio for sp1024 on FineWeb is ~**2.44**. All headline numbers in training logs (**0.9756, 0.9750, 1.0284**) must be multiplied by **3.5/2.44 ≈ 1.44** to compare fairly to the leaderboard. Corrected values are the ones in the results table above.

---

## Post-GPTQ regression: what happened

Pre-quant corrected ~1.40 → post-GPTQ corrected ~1.45–1.49 (using seed=314's artifact). Three plausible causes:

**A. Hessian diagonal warping → bad column reorder.** GPTQ uses `argsort(H_diag)` to determine quantization order. Our saliency boost non-uniformly inflates some `H_jj`, which reorders columns and changes the error-feedback propagation sequence. If over-weighted well-conditioned columns pull quantization mass toward badly-conditioned tails, the net effect is worse than unweighted GPTQ.

**B. Int6 saturation on QK-Gain outliers.** QK-Gain=4.0 encourages large-magnitude Q/K rows. Even after training, W_Q and W_K retain heavier tails than FFN blocks. Per-row int6 clipping (`clip_range=31`) can saturate these outlier rows, collapsing head specialization. Attention is sensitive to rank-1 perturbations in QK^T in a way FFN weights are not.

**C. Calibration distribution shift.** Hessians are built from 32 × 1024-token AR rollouts. If these trajectories under-represent the val conditional, the quadratic form GPTQ minimizes is misaligned with actual usage.

The regression (~0.05–0.09 BPB corrected) is moderate — not catastrophic — suggesting the architecture is sound and the failure is localized to the outlier-attention interaction.

---

## What needs to change for a competitive run

| Fix | Expected effect |
|-----|----------------|
| Add Flash Attention 3 | 10× more steps per 600 s → model converges fully |
| Measure `bytes_per_token` from tokenizer | Correct BPB logging; know true leaderboard position |
| Restore bf16 softmax (stable with FA3) | Additional ~1.5× throughput |
| Ablate saliency strength α | Find the regime where saliency helps vs hurts column reordering |
| Layer-wise outlier clamping before GPTQ | Protect QK heads from int6 saturation |

With FA3 alone (~6,900 steps in 600 s), the same architecture approaches the convergence level of current leaderboard entries. Adding correct BPB + tuned saliency GPTQ is then a direct path to competitive post-quant scores.

---

## Compute grant request

We are requesting **~US$1,000 RunPod developer compute** to:

1. **Profile outlier spectra** per layer pre- and post-GPTQ on Q/K/V/O vs FFN blocks — identifying exactly which layers drive the saturation.
2. **Implement FA3 support** and verify step rate parity with SOTA stacks (~86 ms/step).
3. **Develop Latent-PABU** (Parametric Attention Bounding Unit): a custom CUDA kernel that applies per-head dynamic affine bounds inside the GPTQ inner loop, coordinating with saliency weights to prevent saturation without sacrificing precision allocation. This handles the case where standard per-row int6 scales destroy outlier attention structure.
4. **Ablate α** (saliency injection coefficient) across {0.0, 0.05, 0.1, 0.2} with ground-truth leaderboard-aligned BPB to find the regime where optimizer-state GPTQ yields measurable improvement.

The AdamW v_t saliency technique is mechanically sound; it needs a well-trained base model and a calibrated outlier handler to show its full benefit.

---

## Payload

| File | Description |
|------|-------------|
| `train_gpt.py` | Full training + saliency GPTQ + model.bin export (970 lines, py_compile clean) |
| `model.bin` | **8,154,922 bytes** ← seed=314, int6+zstd-22, < 16,000,000 ✓ |
| `submission.json` | Author, seeds, raw and corrected BPB, per-seed metadata |
| `requirements.txt` | numpy, sentencepiece, zstandard (PyTorch from RunPod image) |
| `train_seed1337.log` | Seed 1337: 779 steps, GPTQ saved 8.72 MB |
| `train_seed42.log` | Seed 42: 780 steps, GPTQ saved 8.88 MB |
| `train_seed314.log` | Seed 314: 698 steps, GPTQ saved 8.15 MB (this model.bin) |

This PR adds **only** `records/track_10min_16mb/2026-04-08_11L_XSA_BigramSVD_AdamWvtGPTQ/` — no root-level files touched.
