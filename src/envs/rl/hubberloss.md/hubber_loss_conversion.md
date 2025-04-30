# Reward Function: Critical Analysis and Proposed Adjustments

## 1. Introduction

The current reward function exhibits extreme skew and heavy tails (kurtosis ≈ 40, skew ≈ 6.4), with 95 % of sampled rewards below 0.085 but outliers up to 0.58. Such a distribution can destabilise training and render most transitions uninformative. This document analyses these issues and proposes concrete modifications to mitigate instability and improve sample efficiency.

## 2. Statistical Observations

- **Mean vs median**: mean = 0.55, median (P50) = 0.04 → a tiny fraction of large rewards inflates the mean.
- **Tail concentration**: P99 = 0.45, P95 = 0.085 → top 1 % of samples account for the bulk of reward mass.
- **Dispersion**: std = 0.063 → relative to mean (CV ≈ 0.11) suggests low overall dispersion, but misleading due to extreme skew.
- **Kurtosis**: ≈ 40 → pronounced heavy‐tailed distribution of rewards.

## 3. Impact on Training

### 3.1 Instability Risk

Rare, large‐reward transitions dominate the TD targets and gradients, causing:

- Oscillations in critic updates when outliers appear or disappear from mini‑batches.
- Over‑fitting to a handful of extreme events, then poor generalisation.

### 3.2 Sample Inefficiency

With 95 % of samples yielding almost zero reward (< 0.085), learning signal is concentrated in 5 % of transitions, slowing convergence and requiring more samples to witness high‐reward events.

## 4. Proposed Modifications

### 4.1 Reward Clipping

Hard‑cap extreme rewards before any scaling.  In the reward lambda, replace raw `r` with a clipped version.

### 4.2 Reward Transformation

Compress outliers with a concave map.  For example, use a signed square‑root or log transform.
**Integrate:**
```python
def transform_reward(raw_r, r_cap=0.12, alpha=10.0):
    """
    Clip, sqrt-transform and log-compress raw reward into [–1, +1].
    """
    t = np.clip(raw_r, -r_cap, r_cap)
    t = np.sign(t) * np.sqrt(np.abs(t))
    t = t / np.sqrt(r_cap)
    t = np.sign(t) * np.log1p(alpha * np.abs(t)) / np.log1p(alpha)
    return float(t)

```

These transforms keep the reward ordering but reduce variance and heavy tails, ensuring more uniform learning signal.

### 4.3 Robust Critic Loss

Apply a concave transform to compress high‐reward outliers while preserving ordering:

- **Square‐root**:
  \(r' = \sqrt{r}\)
- **Logarithm**:
  \(r' = \ln(1 + \alpha r), \quad \alpha>0\)

Transforms reduce variance of the reward signal and distribute it more evenly across transitions.

### 4.3 Robust Critic Loss

Use a Huber loss in place of mean-squared error to reduce sensitivity to large TD errors. Huber combines quadratic behavior for small errors with linear behavior for large ones.

**Huber loss definition (δ = 0.1):**

```python
import jax.numpy as jnp

def huber_loss(td_error, delta=0.1):
    abs_error = jnp.abs(td_error)
    is_small = abs_error <= delta
    small_loss = 0.5 * td_error**2
    large_loss = delta * (abs_error - 0.5 * delta)
    return jnp.where(is_small, small_loss, large_loss)
```

**Integrate into critic update for TD3:**

```python
def critic_update(
    critic_optimiser,
    calculate_td_error_fcn: Callable,
    critic_params: jnp.ndarray,
    critic_opt_state: jnp.ndarray,
    critic_grad_max_norm: float,
    buffer_weights: jnp.ndarray,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_states: jnp.ndarray,
    dones: jnp.ndarray,
    critic_target_params: jnp.ndarray,
    next_actions: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Update TD3 critic networks using clipped double-Q and Huber loss."""
    def loss_fcn(params):
        # compute TD errors via double-Q min target
        td_errors = calculate_td_error_fcn(
            states= jax.lax.stop_gradient(states),
            actions= jax.lax.stop_gradient(actions),
            rewards= jax.lax.stop_gradient(rewards),
            next_states= jax.lax.stop_gradient(next_states),
            dones= jax.lax.stop_gradient(dones),
            critic_params= params,
            critic_target_params= jax.lax.stop_gradient(critic_target_params),
            next_actions= jax.lax.stop_gradient(next_actions)
        )
        # apply Huber per-sample
        loss_per_sample = huber_loss(td_errors, delta=0.1)
        # weight and mean
        weighted_loss = jnp.mean(jax.lax.stop_gradient(buffer_weights) * loss_per_sample)
        return weighted_loss.astype(jnp.float32), td_errors

    # compute gradients and auxiliary TD errors
    (loss_val, td_errors), grads = jax.value_and_grad(loss_fcn, has_aux=True)(critic_params)
    clipped_grads = clip_grads(grads, max_norm=critic_grad_max_norm)
    updates, critic_opt_state = critic_optimiser.update(clipped_grads, critic_opt_state, critic_params)
    critic_params = optax.apply_updates(critic_params, updates)
    return critic_params, critic_opt_state, loss_val, td_errors
```

This modification ensures robust handling of extreme TD errors, integrates double‐Q clipping for TD3, and uses a single loss‐and‐grad evaluation.

### &#x20;4.4 Scaling and Normalisation

Instead of dividing total reward by 10⁵, consider normalising each component:

1. **Normalise error terms** by their dynamic range (e.g. divide (x–x\_ref) by its learned max error).
2. **Normalise final reward** to lie in [0,1] or [–1,1].

This ensures consistent gradient magnitudes across Mach regimes.

### 4.5 Hyperparameter Weight Adjustment

The current exponential weights produce extreme disparities:

- All four reward components use weights in [100,400], leading to near‐zero contributions for small errors but very large spikes for perfect matches.
- Consider reducing weight ratios (e.g. 1–2× range) to smooth the reward surface and encourage moderate improvements rather than all‑or‑nothing.

## 5. Conclusion

By capping extreme rewards, applying nonlinear transforms, and adopting a robust loss, the reward signal becomes more evenly distributed, reducing both instability and sample inefficiency. Normalisation and weight tuning further ensure consistent learning dynamics across trajectory regimes.

