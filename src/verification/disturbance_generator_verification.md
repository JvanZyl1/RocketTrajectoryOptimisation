**Disturbance Generator Implementation**

This document explains the structure and functionality of the AR(1)-based Von Kármán turbulence generator and details the verification tests used to ensure correctness.

---

## 1. VonKarmanFilter (AR(1) Approximation)

The original second-order shaping filter is approximated by a first-order autoregressive process:

- **Time constant**:  
  ω₀ = V / L  
- **Recurrence** (one-step):  
  a = e^(−ω₀ dt),  b = σ √(1 − a²)
- **Derivation**: by discretising an Ornstein–Uhlenbeck Gauss–Markov process dx/dt = −ω₀ x + σ √(2 ω₀) w(t). Exact integration over timestep dt gives xₖ = a xₖ₋₁ + b wₖ, ensuring long-run variance σ² and autocorrelation e^(−ω₀ dt).
- **State update**:
  xₖ = a xₖ₋₁ + b wₖ, where wₖ ~ N(0,1)
- **Output**:
  yₖ = xₖ
- **Reset**: sets x₀ = 0

---

## 2. VKDisturbanceGenerator

Generates body-axis gust forces and moments by combining two independent AR(1) filters.

1. **Parameter sampling**
   - *Length scales* \(L_u,L_v\) uniformly in separate halves of [100, 500] and [30, 300] respectively.
   - *Intensity scales* \(\sigma_u,\sigma_v\) uniformly in separate halves of [0.5, 2.0].

2. **Coupled RNG for scaling tests**
   - Capture NumPy RNG state and filter zero-states at construction.
   - On *first* call with new density \(\rho\), restore these to repeat the same gust sequence.

3. **Output**:
   - Gust velocities: \(u_k, v_k\)
   - Force:  
     \(
       \mathbf{dF} = \tfrac12\,\rho\,V\,[u_k,\,v_k]^{\!T}\,A_
{frontal}
     \)
   - Moment: \(dM = dF_y \times d_{cp\text{-}cg}\)

4. **Reset**:
   - Reseed NumPy’s RNG from the OS and regenerate both filters, ensuring new, decorrelated turbulence.

---

## 3. Verification Tests

| Test Name                         | Purpose                                                    |
|-----------------------------------|------------------------------------------------------------|
| **VonKarmanFilter Reset**         | State is zeroed after `reset()`                           |
| **VonKarmanFilter Zero Noise**    | With `randn()=0`, output & state remain zero               |
| **VonKarmanFilter Unit Noise**    | With `randn()=1`, state & output match `a·0 + b·1`         |
| **VKDisturbanceGenerator Call**   | Returns 2‑d force vector and scalar moment, moment = `dF[1]*d_cp_cg` |
| **VKDisturbanceGenerator Reset**  | After `reset()`, at least one of `L_u, L_v` changes        |
| **Seed Reproducibility**          | Same RNG seed ⇒ identical disturbance sequences            |
| **Reset Decorrelation**           | Sequences before/after `reset()` are uncorrelated (|corr|<0.1) |
| **Long Run Variance**             | `Var(data)/σ² ∈ [0.8,1.2]`                                  |
| **Autocorrelation Timescale**     | Lag‑1 autocorr ≃ \(e^{-ω_0 dt}\) within tolerance 0.1    |
| **PSD Shape**                     | Power spectral density slope in [−2.5, −1.5] (expected –2) |
| **Parameter Ranges**              | `L_u,L_v,σ_u,σ_v` within their prescribed min/max           |
| **Force Moment Scaling**          | Mean force at 2× density ≃ 2× that at 1× density (±10%)    |

Each test is implemented in `disturbance_generator_verification.py` and checks one aspect of the filter or generator behaviour to guarantee compliance with theoretical and functional requirements.

