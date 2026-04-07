# OpenAI Parameter Golf (16MB Challenge) - Test Report & Roadmap

## 1. Codebase Verification & Test Results
Based on `GEMINI.md`, the codebase has undergone a major architectural overhaul, abandoning the Mamba framework in favor of a MOMDP-Transformer (Layer 0 Chassis) consisting of an 11-layer dense transformer with SP1024 tokenization.

The local test suite (`test_nan.py` and `test_nan2.py`) was outdated and still referenced the abandoned `MOMDPMamba` model. I have successfully updated these tests to instantiate the new `GPT` class and `DataLoader`.

**Test Execution Results:**
- **`test_nan.py`**: `PASSED` - Forward pass generates valid logits; no `NaN` values encountered.
- **`test_nan2.py`**: `PASSED` - The `BigramSystem` correctly populated data via SVD-initialization (rank 128) and computed valid projection weights. Forward pass with loss computation generated valid, `NaN`-free loss. 

The Layer 0 chassis is stable and computationally sound.

---

## 2. Execution Roadmap (Strict $25 Cloud Budget)
An 8x H100 instance typically costs between **$25.00 to $40.00 per hour** depending on the cloud provider (e.g., Lambda Labs, RunPod, FluidStack). With a strict $25 budget, we have approximately **45 to 60 minutes** of total compute time. 

Since the official competition constraint is exactly **10 minutes wall-clock training time**, we have enough budget for roughly **4 to 5 full-scale training runs**, provided we don't waste time on environment setup. 

**Rule Zero:** **ABSOLUTELY NO CLOUD DEBUGGING.** All syntax, tensor shapes, and NaN smoke-testing must be done locally (Apple Silicon/CPU) before provisioning the H100s.

### Run 1: "The Layer 1 SOTA Check" (Est. Cost: $5.00)
- **Goal:** Implement the "Validated Frontier Additions" (Layer 1) from `GEMINI.md` and verify we hit the ~1.09 BPB range within 16MB.
- **Implementation tasks (Local):** 
  - AR self-gen calibration for GPTQ int6 quantization.
  - QK-Gain 4.0.
  - Single SLOT vector at the last hidden layer.
- **Cloud Action:** Spin up the 8x H100 instance, clone the repo, run `torchrun` for exactly 10 minutes. Evaluate BPB and artifact size. Immediately terminate instance.

### Run 2: "Layer 2 - Mixture of SLOTs (MoS)" (Est. Cost: $5.00)
- **Goal:** Implement the MoS (K=3 SLOTs + diversity loss) routing.
- **Implementation tasks (Local):** Add the MoS router and diversity loss. Validate backward pass locally to ensure gradients don't break GPTQ quantised backbone.
- **Cloud Action:** 10-minute run. Compare BPB against Run 1.

### Run 3: "Layer 2 - Latent-PABU Integration" (Est. Cost: $5.00)
- **Goal:** Test the Latent-PABU (parallel belief stream) to compensate for XSA self-value removal.
- **Implementation tasks (Local):** Implement the log-depth associative prefix scan. Verify it is mathematically stable.
- **Cloud Action:** 10-minute run. We expect the highest BPB drop here (up to -0.012).

### Run 4: "The Final Assembly & Polish" (Est. Cost: $5.00)
- **Goal:** Combine all successful modules (MoS fed by belief state + Latent-PABU) for the final artifact generation.
- **Cloud Action:** Final 10-minute dash to generate the optimal `model.bin` and `train_gpt.py`.

### Buffer (Est. Cost: $5.00)
- Reserved for unexpected environment setup overhead (e.g., downloading the `fineweb10B_sp1024` dataset to the node, pip installs) or one failed run due to OOM/NCCL timeout. 

## Recommendation
Before we spend any money, we must implement the GPTQ int6 quantization and QK-Gain 4.0 locally (Layer 1). We should only rent the 8x H100 machine when the code is ready for "Run 1".
