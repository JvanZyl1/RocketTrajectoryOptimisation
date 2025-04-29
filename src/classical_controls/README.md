# Classical Control Implementation for Rocket Landing

This document describes the classical control implementations for different phases of rocket landing. Each phase uses specific control strategies optimized for its unique challenges.

## Control Phases Overview

1. Ascent Control
2. Flip-over and Boostback Burn
3. Ballistic Arc Descent
4. Re-entry Burn
5. Landing Burn

## 1. Ascent Control (`ascent_control.py`)

The ascent phase uses a pitch-over maneuver followed by gravity turn control.

### Control Structure
- Initial vertical ascent
- Pitch-over maneuver at predetermined altitude
- Gravity turn following

### Key Parameters
- Pitch-over altitude: \(h_{po}\)
- Initial pitch-over angle: \(\theta_{po}\)
- Pitch rate limit: \(\dot{\theta}_{max}\)

### Control Law
The pitch angle command during gravity turn:
```math
\theta_c = \arctan\left(\frac{v_x}{v_y}\right)
```

## 2. Flip-over and Boostback Burn (`flip_over_and_boostbackburn_control.py`)

Implements a PD controller for the flip-over maneuver and thrust control for the boostback burn.

### Control Structure
- Pitch control using PD with filtering
- Thrust modulation for velocity control

### PD Gains
```python
Kp_theta_flip = -40
Kd_theta_flip = -20
N_theta_flip = 14  # Filter coefficient
```

### Control Laws
Pitch control with filtered derivative:
```math
u_{pitch} = K_p e_\theta + K_d \frac{N}{1 + N\frac{s}{\omega_c}} \dot{e}_\theta
```
where ```math e_\theta = \theta_{desired} - \theta_{actual}```

## 3. Ballistic Arc Descent (`ballisitic_arc_descent.py`)

Passive phase with minimal control, mainly for aerodynamic stability.

### Control Structure
- Angle of attack control for stability
- No active thrust control

### Control Law
Angle of attack command:
```math
\alpha_c = K_\alpha \cdot \text{sign}(v_\infty) \cdot \min(|\alpha_{max}|, |\alpha_{req}|)
```

## 4. Re-entry Burn (`re_entry_burn.py`)

Complex control system managing both attitude and velocity during atmospheric re-entry.

### Control Structure
- Throttle control based on dynamic pressure
- Pitch control using PD with filtering
- Thrust vector control through gimbal angle

### Gains and Parameters
```python
Kp_mach = 0.049      # Throttle control gain
Q_max = 30000        # Maximum dynamic pressure [Pa]
M_max = 0.75e9       # Maximum control moment
```

### Control Laws

Throttle control:
```math
Q_{ref} = Q_{max} - 1000 \text{ Pa}
```
```math
M_{ref} = \sqrt{\frac{2Q_{ref}}{\rho}} \cdot \frac{1}{a}
```
```math
u_{throttle} = K_p(M_{ref} - M)
```

Pitch control:
```math
\alpha_{eff} = \gamma - \theta - \pi
```
```math
M_z = K_p \alpha_{eff} + K_d \frac{N}{1 + N\frac{s}{\omega_c}} \dot{\alpha}_{eff}
```

Gimbal angle determination:
```math
\delta = \arcsin\left(\frac{-M_z}{T_g d}\right)
```
where:
- ```math T_g``` is the gimballed thrust
- ```math d``` is the distance from thrust point to CG

## 5. Landing Burn (`landing_burn.py` and `landing_burn_optimise.py`)

Two-phase control system with optimization for fuel efficiency.

### Control Structure
- Initial phase: Position error reduction
- Terminal phase: Zero velocity at touchdown

### Optimization Parameters
- Burn start time
- Initial thrust vector
- Thrust profile

### Control Laws
Position error:
```math
e_p = \begin{bmatrix} x_{target} - x \\ y_{target} - y \end{bmatrix}
```

Thrust commands:
```math
\begin{bmatrix} T_x \\ T_y \end{bmatrix} = K_p e_p + K_d \begin{bmatrix} \dot{x} \\ \dot{y} \end{bmatrix}
```

## Optimization Results

The controllers have been tuned using various optimization techniques, with results shown in:
- `landing_burn_optimisation_history.png`
- `re_entry_burn_optimisation_history.png`
- `flip_over_and_boostbackburn_pitch_tuning.png`

## Simulation Results

Detailed simulation results are available in the `results/classical_controllers` directory:
- Trajectory plots
- State variable histories
- Control input histories