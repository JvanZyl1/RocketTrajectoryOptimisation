import math
import numpy as np


landing_prop_percentage_stage_1 = 0.98
landing_prop_percentage_stage_2 = 0.98

# Stage 1 : Super Heavy
m_stage_1 = 3675e3 # kg
m_prop_1 = 3400e3 # kg
m_strc_1 = 275e3 # kg

# Stage 2 : Starship
m_stage_2 = 1600e3 # kg
m_prop_2 = 1500e3 # kg
m_strc_2 = 100e3 # kg

# Stage 3 : Payload
m_payload = 100e3 # kg

# Convert to useful masses
m_initial = m_stage_1 + m_stage_2 + m_payload
m_stage_1_burn_out = m_initial - m_prop_1 * landing_prop_percentage_stage_1
m_stage_1_separation = m_stage_1_burn_out - m_strc_1
m_stage_2_burn_out = m_stage_1_separation - m_prop_2 * landing_prop_percentage_stage_2

# Rocket parameters
D_rocket = 9 # [m]
S_rocket = np.pi * (D_rocket/2)**2 # [m^2]
from TrajectoryGeneration.drag_coeff import compile_drag_coefficient_func
get_drag_coefficient_func_stage_1 = compile_drag_coefficient_func(alpha = 5)
get_drag_coefficient_func_stage_2 = compile_drag_coefficient_func(alpha = 5)

# Engine parameters
Isp_stage_1 = 350 # [s]
Isp_stage_2 = 380 # [s]

v_ex_stage_1 = Isp_stage_1 * 9.81 # [m/s]
v_ex_stage_2 = Isp_stage_2 * 9.81 # [m/s]

T_engine_stage_1 = 2745e3 # [N]
T_engine_stage_2 = 2000e3 # [N]
n_engine_stage_1 = 33
n_engine_stage_2 = 6

T_max_stage_1 = T_engine_stage_1 * n_engine_stage_1 # [N]
T_max_stage_2 = T_engine_stage_2 * n_engine_stage_2 # [N]

m_dot_stage_1 = T_max_stage_1 / v_ex_stage_1 # [kg/s]
m_dot_stage_2 = T_max_stage_2 / v_ex_stage_2 # [kg/s]

t_burn_stage_1 = m_prop_1 / m_dot_stage_1 # [s]
t_burn_stage_2 = m_prop_2 / m_dot_stage_2 # [s]

nozzle_exit_area = 1.326 # [m^2]
nozzle_exit_pressure_stage_1 = 100000 # [Pa]
nozzle_exit_pressure_stage_2 = 100000 # [Pa]

### Orbital parameters ###
h_vertical_rising = 100 # [m]
h_gravity_turn = 100e3 # [m] but go to max 100e3

kick_angle = math.radians(-0.8) # [rad] : so ends at roughly 45 degrees flight path angle.

throttle_gravity_turn = 0.95
h_throttle_gt_0 = 5000 # [m]
h_throttle_gt_1 = 20000 # [m]

t_coast_endo = 5 # [s]
