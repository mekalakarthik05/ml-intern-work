# Scaling Laws

**Bigger models, more data, more compute — predict exactly how performance improves before you spend a dollar.**

You will learn how neural network performance follows predictable power-law trends as you scale model size, dataset size, and compute. These laws let you estimate the optimal resource allocation for training a model before writing a single line of training code.

---

## What Are Scaling Laws?

Scaling laws describe a striking empirical finding: the test loss of a neural network decreases as a **power law** (straight line on a log-log plot) when you increase the number of parameters, the dataset size, or the compute budget — over many orders of magnitude. This was systematically documented by Kaplan et al. (2020) for language models and later refined by Hoffmann et al. (2022, "Chinchilla").

Think of it like building a house. Doubling the blueprint detail (parameters) helps, but only if you also double the building materials (data) and the construction time (compute). Neglect any one, and the others saturate in usefulness. Scaling laws tell you exactly how much of each you need to hit your target performance.

The key insight: **performance is bottlenecked by the smallest resource.** If you have a 7B-parameter model but only 1B tokens of data, you have wasted most of the model's capacity. Conversely, 1T tokens of data with a 100M-parameter model means most of the data never translates into better performance. The optimal allocation — where model size and data size are balanced — is called **compute-optimal training**, and the Chinchilla paper showed that most large models were significantly undertrained (too many parameters for the amount of data).

```mermaid
flowchart LR
    subgraph Inputs
        N[Model Parameters N]
        D[Dataset Tokens D]
        C[Compute FLOPs C]
    end
    subgraph Scaling Law
        L[Test Loss L]
    end
    N -->|L ∝ N^{-α_N}| L
    D -->|L ∝ D^{-α_D}| L
    C -->|L ∝ C^{-α_C}| L
```

---

## Mathematical Formulation

| Concept | Equation | What it tells us |
|---------|----------|------------------|
| Kaplan et al. power law (params) | $L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N} + L_\infty$ | Loss scales as a power of model size; doubling N reduces loss by a fixed fraction determined by $\alpha_N \approx 0.076$ |
| Kaplan et al. power law (data) | $L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty$ | Loss scales as a power of dataset size; $\alpha_D \approx 0.095$ for language models |
| Compute budget | $C \approx 6 \cdot N \cdot D$ | Each training step costs roughly $6ND$ FLOPs (forward + backward pass). This ties parameters, data, and compute together |
| Chinchilla optimal allocation | $N_{\text{opt}} \propto C^{0.5},\ D_{\text{opt}} \propto C^{0.5}$ | For a given compute budget, parameters and data should be scaled equally — doubling compute means 26% more parameters and 26% more data (not 50% more of one) |
| Irreducible loss | $L_\infty$ | The floor below which loss cannot fall — represents the entropy of the data distribution itself |
| Cross-entropy to FLOPs | $L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}$ | Performance as a function of total compute budget, combining the effects of $N$ and $D$ |

**Key exponents** (Chinchilla, 2022): $\alpha_N \approx 0.34$, $\alpha_D \approx 0.28$, $\alpha_C \approx 0.05$ under the parametric fit. These differ from Kaplan because of different fitting methodology and the inclusion of the FLOPs constraint.

---

## How It Works — Step by Step

**Step 1 — Train models at multiple scales.** Train 8–10 models with increasing parameter counts (e.g., 10M, 30M, 100M, 300M, 1B) on progressively larger subsets of data. Record the final test loss for each.

*Analogy:* Bake cakes with 1, 2, 4, and 8 eggs, measuring how tall each rises.

**Step 2 — Plot loss vs. size on a log-log scale.** You should see (approximately) a straight line. Fit a power law $L(N) = \alpha N^{-\beta} + L_\infty$ using least-squares on log-transformed data.

*Analogy:* The relationship "height rises like (eggs)^0.3" — predictable and smooth.

**Step 3 — Estimate the irreducible loss $L_\infty$.** This is the asymptotic floor — the loss you would get with infinite data and parameters, determined by the intrinsic entropy of the data.

**Step 4 — Repeat for data scaling.** Keep model size fixed and vary dataset size. Fit $L(D) = \gamma D^{-\delta} + L_\infty$. The exponents $\beta$ (parameter scaling rate) and $\delta$ (data scaling rate) tell you which resource is more valuable at the margin.

**Step 5 — Compute-optimal planning.** Given a target loss and a compute budget, solve $C = 6ND$ with $L(N, D) = L_\infty + (N_c/N)^{\alpha_N} + (D_c/D)^{\alpha_D}$ to find the optimal $N$ and $D$.

*Analogy:* You have a fixed budget for eggs and flour. Scaling laws tell you exactly how many of each to buy so neither ingredient is wasted.

---

## Key Assumptions

| Assumption | If violated |
|------------|-------------|
| Training is not bottlenecked by hyperparameters (LR, batch size, schedule must be optimal at each scale) | Suboptimal training at small scales produces a steeper-than-expected slope, overestimating the benefit of scale |
| The model architecture is fixed across scales (same depth/width ratio, same activation function) | Changing architecture breaks the power law — you must re-estimate exponents for each architecture family |
| Data distribution is identical at all scales (subsets are randomly sampled) | If small-scale data is higher quality than large-scale data, the power law overestimates the value of more data |
| Irreducible loss $L_\infty$ is constant | For noisy data sources, $L_\infty$ may itself improve with scale (better deduplication, higher-quality filtering) |
| The power law holds beyond the observed range | Extrapolation is risky — at extreme scales, new phenomena emerge (memorization, capability jumps) that violate smooth power laws |

---

## When to Use / When Not to Use

| Use Scaling Laws When ... | Avoid Them When ... |
|---------------------------|---------------------|
| You are planning a large training run (>$10k GPU-hours) and need to budget resources | Your model is smaller than ~10M parameters (power laws are noisy at very small scales) |
| You need to decide between training a bigger model or collecting more data | Your architecture or data distribution changes between experiments |
| You want to predict the performance of a model before training it | You need exact predictions (scaling laws give estimates within ±0.1 nats, not exact values) |
| You are comparing different model families (dense vs MoE, different tokenizers) | Your training is compute-unbounded (no resource constraint to optimize for) |

---

## Implementation Overview

| Aspect | From Scratch (NumPy) | Library (scipy / sklearn) |
|--------|----------------------|---------------------------|
| Fit power law | Manual log-transform + least squares: `np.polyfit(np.log(N), np.log(L - L_inf), 1)` | `scipy.optimize.curve_fit` with custom `L(N, L_inf, alpha, N_c)` function |
| Compute-optimal search | Grid search over N and D satisfying $6ND = C_{\text{total}}$ | `scipy.optimize.minimize` with Chinchilla's parametric loss |
| Irreducible loss $L_\infty$ | Extrapolate from largest models by fitting $L(N) = aN^{-b} + c$ | Wrapped in `curve_fit` as a free parameter with bounds |
| Exponent estimation | Bootstrapped sampling to get confidence intervals on $\alpha$ and $\beta$ | `sklearn.linear_model.LinearRegression` on log-transformed data |

```python
# Estimate scaling exponent from observed losses at different model sizes
import numpy as np
from sklearn.linear_model import LinearRegression

N = np.array([85e6, 303e6, 1.1e9, 3.9e9])           # parameter counts
L = np.array([4.12, 3.67, 3.32, 3.08])              # observed losses
log_N = np.log(N).reshape(-1, 1)


reg = LinearRegression().fit(log_N, np.log(L))
alpha_N = -reg.coef_[0]                              # power-law exponent ≈ 0.076
print(f"Scaling exponent alpha_N = {alpha_N:.3f}")
```

---

## Top 5 Interview Questions

**Q1. What is a scaling law and why does it hold?**
- Power-law decay of loss with N, D, or C. Hypothesized causes: (a) smooth data manifold with power-law spectral density, (b) model capacity as a resource like any other diminishing-returns input. Mention Kaplan (2020) vs Chinchilla (2022) as key references.

**Q2. How did Chinchilla differ from Kaplan?**
- Kaplan: N and D should be scaled unevenly (more N than D). Chinchilla: N and D must be scaled equally ($N_{\text{opt}} \propto C^{0.5}$, $D_{\text{opt}} \propto C^{0.5}$). Reason: Kaplan didn't fix compute budget; Chinchilla fixed C and found the optimal allocation. Most pre-2022 models were overparametrised and undertrained.

**Q3. If scaling laws are power laws, why did GPT-4 seem like a "jump"?**
- Scaling laws predict smooth improvement in pre-training loss, not in downstream capabilities. Emergent abilities (reasoning, code generation) appear as sharp phase transitions on specific benchmarks even as pre-training loss improves smoothly. This is a measurement artifact: the wrong metric was being tracked.

**Q4. How would you estimate the scaling exponent from scratch?**
- Train 8+ models at logarithmically spaced sizes, record final validation loss, subtract estimated $L_\infty$, take log of both axes, fit a line. Use bootstrap to get confidence intervals. Validate by training 2 models at the largest scale to check reproducibility.

**Q5. What happens when you violate the "fixed architecture" assumption?**
- Changing depth/width ratio, activation function, or optimizer changes the exponent. You must re-estimate. Example: GPT-3 used a different depth/width ratio than Chinchilla — different exponents. Mixture-of-Experts models have different scaling behaviour (shallower slope because MoE increases parameters without proportional compute).

---

## Quick Reference Table

| Item | Detail |
|------|--------|
| Algorithm Type | Empirical law (not an algorithm) — used for resource planning |
| Training Paradigm | Estimates from supervised pre-training runs at multiple scales |
| Time Complexity | $O(M \cdot C_{\text{train}})$ for fitting — where $M$ is the number of training runs (typically 8–16) and $C_{\text{train}}$ is the cost of each run |
| Space Complexity | Negligible (storing a few dozen loss values) |
| Key Hyperparameters | Exponent $\alpha_N$, $\alpha_D$, irreducible loss $L_\infty$, compute budget $C$ |
| Evaluation Metrics | Cross-entropy loss (validation), FLOPs-adjusted loss, $\chi^2$ goodness-of-fit of power law |

---

## References & Further Reading

1. **Kaplan et al. (2020):** *Scaling Laws for Neural Language Models* — the original paper that established power-law scaling for transformers
2. **Hoffmann et al. (2022):** *Training Compute-Optimal Large Language Models* — Chinchilla paper that corrected the optimal allocation
3. **Best tutorial:** *A Brief Introduction to Scaling Laws* by Jaime Sevilla (2023) — blog post with interactive visualisations
4. **Kaggle notebook:** *Scaling Laws Exploration* — search for "scaling laws LLM" on Kaggle to find notebooks fitting power laws to publicly available loss curves
5. **Scaling in vision:** Zhai et al. (2022) — *Scaling Vision Transformers* — shows similar power laws for image models
