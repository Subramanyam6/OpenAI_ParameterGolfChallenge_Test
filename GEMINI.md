# Project Context: OpenAI Parameter Golf (16MB Challenge)

## 1. The Core Mission
- **Objective:** Achieve the lowest Bits Per Byte (BPB) score on the FineWeb-Edu validation set.
- **Hard Constraints:**
  - 16,000,000 byte limit (decimal) for the final `.zip` artifact (must include `train_gpt.py` + `model.bin`).
  - 10 minutes total wall-clock training time on 8x H100 GPUs.
- **Competition State (April 2026):**
  - Official SOTA: **1.1147 BPB** (#1019, XSA-all + GPTQ + AR Self-Gen)
  - Best pending pure-neural: **1.0914 BPB** (#1176, QK-Gain 4.0 + SLOT + Muon-TTT)
  - Best pending overall: **0.4027 BPB** (#1094, Causal BackoffNgramMixer — n-gram track)
  - Baseline: **1.2244 BPB**

---

## 2. Abandoned Approaches (Empirically Dead)

The following were considered but confirmed non-viable by competition data:

- **Mamba / SSMs at dim=512:** Tested at competition scale (#1227) — +2.7% BPB *worse* than transformer. Catastrophic with GPTQ quantization.
- **Depth recurrence ≥ 3 cycles:** Quantization error amplifies ~900× (#363). 8-cycle default = certain divergence.
- **Byte-level vocabulary (size=256):** ~4–6× longer sequences → ~16× attention FLOPs → training never converges in 600s.
- **Entropy-triggered TTT:** 7 failed variants tested at frontier. Neutral to negative. Entropy gating does not improve BPB, only saves eval time.
- **System-1-Discounted Loss (Complementary Training):** +2.6% BPP at full scale (#1227) without a strong eval-time n-gram system. Our 1MB bigram cannot compensate.

---

## 3. Current Architecture: MOMDP-Transformer (Layer 0 Chassis)

**Philosophy unchanged — execution overhauled.** We retain the MOMDP framing (System 1 fast reflex + System 2 deep belief) but implement it on the empirically validated transformer chassis, not Mamba.

### Layer 0 (Implemented & Tested)

| Component | Choice | Rationale |
|---|---|---|
| Architecture | 11-Layer Dense Transformer | Competition consensus; depth beats width at this scale |
| Tokeniser | SP1024 | 4–6× fewer tokens than byte-level → more training steps |
| d_model / n_heads / MLP | 512 / 8 / 3× (hidden=1536) | MLP 3× funded by int6 artifact savings |
| seq_len | 1024 | Competition consensus |
| Positional encoding | RoPE (all layers) | Standard |
| Tied embeddings | `lm_head.weight = embed.weight` | Saves ~1 MB in artifact |
| XSA | All 11 layers | Removes self-value bias; ~0.006 BPB; zero params |
| Activation | relu² | Works with XSA; natural sparsity helps int6 |
| Optimizer | Parallel Muon (2D) + AdamW (1D/0D) | Competition consensus; lr=0.02, WD=0.04 |
| Momentum warmup | 0.95 → 0.99 over 1500 steps | Consensus |
| EMA | decay=0.997 | Requires XSA to be positive; eval always uses EMA model |
| System 1 | BigramSystem: embed(2048,128)→proj(128,1024) | Stats-initialized via `index_add_` (GPU-native; avoids numpy copy-discard bug) |
| Eval | Sliding window stride=64, window=1024 | Worth ~0.034 BPB vs non-overlapping |
| Batch | 786K tokens | Competition frontier consensus |
| Schedule | Warmup 750 / Cooldown 3000 steps | Consensus |

**Params (deduplicated):** 29,765,120 | **FP32 size:** 119.1 MB | **Post-GPTQ int6+zstd-22 estimate:** ~14–15 MB (fits 16 MB)

### Layer 0 Local Test Results (Apple Silicon MPS, dummy random data)
- Smoke test: **PASSED** — no NaN in loss, gradients, or EMA; 5 training steps complete
- Mini-run (60 steps): Step 0 BPB=4.23, Step 30 BPB=4.19 (both above random baseline 2.857 — correct for untrained model on random data)
- `model.bin` saves correctly
- DDP code path structured for `torchrun --nproc_per_node=8` (NCCL path not tested locally — standard PyTorch DDP)

### Bugs Fixed Before H100 Launch
1. **BPB formula was wrong**: `loss / log(2)` gave bits/token not bits/byte. Fixed to `loss / log(2) / bytes_per_token` (default 3.5 for SP1024 on English text; override via `BYTES_PER_TOKEN` env var).
2. **DDP val BPB was rank-0-only**: Added `dist.all_reduce` so all 8 ranks contribute to the reported BPB.
3. **Step 0 never evaluated**: Removed `step > 0` gate; initial BPB now logged as baseline reference.

### H100 Launch Checklist (do in order)
1. Confirm data path: `/data/datasets/fineweb10B_sp1024/fineweb_{train,val}_*.bin` exists on RunPod.
2. **Sanity run first (costs ~$0.50):** `ITERATIONS=500 torchrun --nproc_per_node=8 train_gpt.py` — verify NCCL init, DDP all_reduce, non-NaN BPB above ~2.8.
3. Check that `bytes_per_token` is accurate (measure from fineweb: `avg(utf8_bytes_per_sequence) / seq_len`). Override with `BYTES_PER_TOKEN=X` if needed for accurate leaderboard comparison.
4. **Full run:** `torchrun --nproc_per_node=8 train_gpt.py` — targets <1.115 BPB after GPTQ.

---

## 4. Remaining Implementation Layers

### Layer 1 — Validated Frontier Additions (next step, H100 required)

| Technique | Expected BPB delta | Notes |
|---|---|---|
| GPTQ int6 (AR self-gen calibration, within 600s) | Unlocks full 16MB budget | Largest missing piece |
| QK-Gain 4.0 | −0.004 BPB | Competition-confirmed (#1176) |
| Legal backward-looking TTT (score-first AdamW) | −0.003 to −0.008 BPB | NEVER adapt then rescore same tokens |
| Single SLOT vector at last hidden layer | −0.002 to −0.006 BPB | Ablation baseline for Layer 2 |

**Projected BPB after Layer 1: ~1.09–1.095 BPB**

### Layer 2 — Novel Untested Proposals

| Technique | Expected BPB delta | Rationale |
|---|---|---|
| MoS (K=3 SLOTs + diversity loss) | −0.003 to −0.007 BPB | SLOT gradients stay outside GPTQ-quantized backbone; STE winner-take-all routing; diversity loss prevents router collapse |
| Latent-PABU (parallel belief stream) | −0.003 to −0.012 BPB | XSA removes self-value info; PABU provides a parallel channel exempt from XSA; implemented via log-depth associative prefix scan (NOT torch.cumprod alone) |
| MoS router fed by b_t | −0.002 to −0.005 BPB | Belief state stabilises router, solves cold-start; partially additive with above |

**Projected BPB after Layer 2: ~1.073–1.085 BPB** (probability-weighted estimate; novel, unvalidated)

---

## 5. What NOT to Build
- MoE at this scale: −0.06 to −0.08 BPB vs dense (#480, Apple scaling laws confirm optimal sparsity = 0 below ~500M params)
- INT4 quantization: +0.065 BPB penalty (#480)
- Multi-epoch TTT (>3 epochs): memorisation, not compression (#568: 5 epochs → 0.78 BPB = near-certain data reproduction)
- Knowledge distillation: +0.003 BPP at all alpha values in competition test (#1029)
- 2:4 structured sparsity: +0.672 BPB (#1105)

---

## Progress Log
- **[Step 1]** Initial Setup: Cloned `openai/parameter-golf`. Analysed baseline.
- **[Step 2]** Mamba architecture scaffold (abandoned — empirically dead at scale).
- **[Step 3]** Mamba implementation with MOMDP framing. Multiple critical bugs: Python loop for SSM (100× too slow), depth recurrence × 8 (catastrophic), byte-level vocab, rules-violating TTT, numpy copy-discard bug in bigram.
- **[Step 4]** Full competitive analysis: compared against competition leaderboard data, identified all failure modes.
- **[Step 5]** **LAYER 0 COMPLETE:** Full rewrite of `train_gpt.py` as 11L Transformer chassis. Local smoke test PASSED on Apple Silicon MPS. Ready for 8×H100 full run via `torchrun --nproc_per_node=8 train_gpt.py`.
- **[Step 6]** **PRE-H100 AUDIT:** Fixed 3 bugs (BPB formula bits/token→bits/byte, DDP val BPB all_reduce missing, step-0 eval disabled). Verified no train/val data contamination. BPB formula confirmed correct (4.23 bits/byte on random data with untrained model, vs 2.857 random baseline — correct direction). SVD on CUDA will run natively (no MPS fallback).
