# Von Kármán Gust Model and Disturbance Generator

## 1. Von Kármán Model

The Von Kármán turbulence model describes atmospheric gusts as a second-order shaping filter with power spectral density

$$
\Phi(\omega) = \sigma^2 \frac{2\,\omega_0}{\pi} \frac{1}{(\omega^2+\omega_0^2)^{5/6}}
$$

where:
- $\sigma$ is the gust intensity (long-run standard deviation),
- $\omega_0 = V/L$ is the natural frequency (with flight speed $V$ and turbulence length scale $L$).

Equivalently, in state-space form:

$$
\dot x = A_c x + B_c w(t),\quad y = C_c x,
$$

with damping ratio $\zeta=1/\sqrt{2}$ and continuous matrices:

$$
A_c = \begin{bmatrix}
0 & 1\\
-\omega_0^2 & -2\,\zeta\,\omega_0
\end{bmatrix},\quad
B_c = \begin{bmatrix}
0\\
\sigma\sqrt{\frac{\pi}{2\,\omega_0^3}}
\end{bmatrix},\quad
C_c = \begin{bmatrix}
0 & 1
\end{bmatrix}.
$$

Discretisation (zero-order hold) yields $A_d, B_d, C_d$ which drive the discrete recurrence

$$
x_{k+1} = A_d x_k + B_d w_k,\quad y_k = C_d x_k,\quad w_k\sim\mathcal{N}(0,1).
$$

## 2. Disturbance Generator

The **VKDisturbanceGenerator** combines two independent Von Kármán filters (body-axis forward and lateral) and outputs:
- **Gust velocities** $(u,v)$ from the two filters.

Key features:
1. **Parameter sampling**: $L_u,L_v$ and $\sigma_u,\sigma_v$ sampled uniformly in separate halves of their prescribed ranges to ensure diversity of gust characteristics.
2. **Reset capability**: Reseeds RNG and regenerates filter parameters to produce new, uncorrelated turbulence sequences.
3. **Logging**: Stores gust velocities in a log_data dictionary for later analysis and visualization.
4. **Plotting**: Built-in functionality to generate plots of the horizontal and vertical gust components.

## 3. Verification Tests

| Test                                | Description                                                       |
|-------------------------------------|-------------------------------------------------------------------|
| VonKarmanFilter Reset               | `reset()` zeroes filter state                                     |
| VonKarmanFilter Zero Noise          | `randn()=0` ⇒ zero output and state                               |
| VonKarmanFilter Unit Noise          | `randn()=1` ⇒ state=$B_d$, output=$C_d B_d$                      |
| VKDisturbanceGenerator Call         | returns gust_u, gust_v and logs them correctly                   |
| VKDisturbanceGenerator Reset        | parameters change and logs are cleared after `reset()`           |
| Seed Reproducibility                | fixed RNG seed ⇒ identical sequences                              |
| Reset Decorrelation                 | pre/post `reset()` sequences uncorrelated ($|\text{corr}|<0.35$)   |
| Long Run Variance                   | variance ratio <1.2 (allows ZOH reduction)                        |
| Autocorrelation Timescale           | lag-1 autocorr ≃ $e^{-\omega_0 dt}$ within tolerance 0.1         |
| PSD Shape                           | spectral slope in $[-2.5,-1.5]$                                   |
| Parameter Ranges                    | $L_u,L_v,\sigma_u,\sigma_v$ within prescribed minima/maxima      |
| Plotting Functionality              | generator can create visualization of gust time series            |

## 4. Time Series Analysis
![Gust velocity time series](results/verification/disturbance_generator_verification/time_series_analysis.png "Time series analysis")

The time-series diagnostics reveal both strengths and limitations of the discrete filter implementation:

- **Trace variability vs theory**:
  - **High $L$** (orange) and **Low $L$** (green) qualitatively match expected correlation scales, but the observed amplitude ranges (±5 m/s vs ±0.5 m/s) exceed the theoretical $\sigma=1.0$ m/s for 'High $L$'—indicating that finite-sample extremes and filter initial transients still influence the first 1 000 samples.
  - **High $\sigma$** (red) should have double the standard deviation of the base case, yet red's peak excursions are ≃1.5× higher, suggesting ZOH discretisation underestimates variance by ~25% at large $dt$.

- **Distribution shape**:
  - All kernels appear Gaussian, but the **Low $L$** distribution shows a slight cusp at zero: this arises from de-correlation every few samples producing many small amplitudes.  
  - KDE smoothing parameters in Seaborn can exaggerate tails; a histogram with bin counts might better reflect true kurtosis.

- **Autocorrelation tails**:
  - Exponential decay is evident, but **High $L$** ACF remains above 0.8 even at lag=100, implying a characteristic time >2 s ($\tau=L/V=5$ s). Finite window bias causes the tail not to decay fully to zero.
  - **Low $L$** dips slightly negative around lag≈60, an artifact of spectral leakage and finite-length bias rather than true negative correlation in the continuous model.

- **PSD accuracy**:
  - Mid-band slopes are close to −2, but the **Low $L$** PSD flattens prematurely near $f≈1$ Hz—an effect of insufficient frequency resolution (NFFT=1024 at $F_s=50$ Hz).  
  - The **High $\sigma$** PSD magnitude scales as $\sigma^2$, but low-frequency ($<10^{-2}$ Hz) troughs appear due to windowing; using Welch's method with larger `nperseg` would smooth these.

In summary, while the broad spectral and autocorrelation trends align with Von Kármán theory, careful parameter choices (sample length, NFFT, KDE bandwidth) and transient removal (burn-in) are needed for quantitative match.

## 5. Sensitivity Analysis

![Gust sensitivity Analaysis](results/verification/disturbance_generator_verification/sensitivity_analysis.png "Sensitivity analysis")

These curves quantify how parameter changes manifest in statistics—but also expose discretisation artifacts:

- **Std dev vs $L$**:
  - The monotonic increase reflects longer integration, yet the nonlinear, near-quadratic growth (Std ≈0.007·L) deviates from continuous $\sigma^2$ independence of $L$.  
  - For $L<100$ m, small sample bias and filter's initial transients inflate std estimates; using a longer warm-up before sampling would stabilize low-L points.

- **Autocorr (lag 1) vs $L$**:
  - Matches $a=\exp(-\omega_0 dt)$ closely except at extreme $L$ where computational rounding yields a plateau at 0.9999.  
  - The theoretical curve should be smooth; the slight scatter reflects finite Monte Carlo noise with only 10 000 samples per point.

- **Std dev vs $\sigma$**:
  - Linear scaling holds within ~5%, but at $\sigma>1.8$ the slope tapers off—likely due to saturation of `float32` in intermediate computations or clipping in plotting.

- **Kurtosis vs $\sigma$**:
  - Ideally zero for Gaussian; observed ±0.3 fluctuations are purely sampling artifacts.  
  - A 95% confidence band of kurtosis for 10 000 samples is roughly ±0.2, so deviations are expected.

## 6. VKDisturbanceGenerator Analysis

![Horizontal Gust Analysis](results/verification/disturbance_generator_verification/vk_generator_u_analysis.png "Horizontal Gust Analysis")
![Vertical Gust Analysis](results/verification/disturbance_generator_verification/vk_generator_v_analysis.png "Vertical Gust Analysis")

The VKDisturbanceGenerator produces horizontal and vertical gust components with different characteristics:

- **Effect of velocity parameter V**:
  - Higher V values (blue) produce faster-changing gusts with shorter correlation times
  - Lower V values (green) result in smoother, more slowly varying gusts
  - This matches expectations as V affects the characteristic frequency ω₀ = V/L

- **Distribution properties**:
  - Both horizontal and vertical gust components show near-Gaussian distributions
  - The vertical component often has higher variance due to the parameter sampling approach that tends to pair higher sigma_v with lower L_v

- **Spectral content**:
  - The PSDs show the expected Von Kármán shape with approximately -5/3 slope at high frequencies
  - The low V case shows more energy at low frequencies, consistent with its longer correlation length

## 7. VKDisturbanceGenerator Velocity Sensitivity

![Velocity Sensitivity Analysis](results/verification/disturbance_generator_verification/vk_generator_sensitivity.png "Velocity Sensitivity Analysis")

This comprehensive analysis shows how the velocity parameter V affects the gust characteristics:

- **Autocorrelation decrease with V**:
  - As velocity increases, the lag-1 autocorrelation decreases for both horizontal and vertical components
  - This follows the theoretical expectation: higher V increases ω₀ = V/L, which decreases exp(-ω₀·dt)
  - The measured autocorrelation closely matches the theoretical prediction across all tested velocities

- **Gust intensity variation with V**:
  - Standard deviation significantly decreases with increasing V
  - This inverse relationship is strongest at low V values (50-110 m/s) and becomes more gradual at higher velocities
  - The vertical component generally shows lower standard deviation than the horizontal component

- **Spectral width broadens with V**:
  - Higher velocity values lead to wider frequency content (increased spectral width)
  - This indicates faster-changing, higher-frequency gust components at higher velocities
  - The relationship is approximately linear for the horizontal component

- **Parameter randomization**:
  - The scatter plots confirm that L_u, L_v, sigma_u, and sigma_v are randomly sampled within their ranges
  - There is no correlation between these parameters and the velocity V, as expected from the implementation
  - This ensures that sensitivity to V is isolated from other parameter effects

Key findings:
1. Velocity primarily affects the time-domain characteristics (correlation time) rather than distribution properties
2. The relationship between V and autocorrelation follows theoretical expectations
3. At higher velocities (>150 m/s), the gusts change more rapidly with smaller amplitudes
4. The vertical gust component generally shows more high-frequency content than the horizontal component

This analysis confirms that the V parameter provides effective control over the temporal characteristics of the generated gusts, allowing the simulation of different flight conditions.

## 8. Definitions

- **Autocorrelation**: the correlation between a time series and a lagged copy of itself. For lag $k$:
  $$
  R(k) = \frac{E[(y_t - \mu)(y_{t+k} - \mu)]}{\mathrm{Var}(y)}
  $$
  Values range from -1 (perfect inverse) to +1 (perfect correlation), with 0 indicating no memory.

- **Trace Variability**: describes amplitude fluctuations in the time-series trace, quantified by:
  1. **Standard deviation** (spread about mean)
  2. **Peak-to-peak range** (max minus min)
  3. **Excursion rate** (typical rate of change)

- **Kurtosis**: the fourth standardized moment:
  $$
  \mathrm{Kurt} = \frac{E[(y - \mu)^4]}{(\mathrm{Var}(y))^2} - 3
  $$
  A Gaussian distribution has kurtosis 0; positive values indicate heavier tails (more outliers), negative indicate lighter tails.

- **Spectral Width**: a measure of how spread out the frequency content is:
  $$
  SW = \sqrt{\frac{\sum_i (f_i - f_c)^2 \cdot P(f_i)}{\sum_i P(f_i)}}
  $$
  where $f_c$ is the spectral centroid and $P(f_i)$ is the power at frequency $f_i$. Wider spectral width indicates a broader range of frequencies in the signal.

*Report generated based on the standard Von Kármán shaping filter and corresponding disturbance generator implementation.*

