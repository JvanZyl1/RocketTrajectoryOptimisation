clear all
close all
clc


m_0 = 6492 * 1000;
m_p_0 = 4165 * 1000;

d_tcg = 60;

g0 = 9.81;

T_e = 2745*1000;    % Engine thrust [N]
n_eg_g = 3 + 12;    % Number of gimballed engines
n_eg_ng = 26;       % Number of non-gimballed engines
p_e = 100000;       % Nozzle exit pressure [Pa]
p_a = 101325;       % Ambient pressure [Pa]
A_e = 0.01;         % Engine nozzle exit area [m^2]

T_e_with_losses = T_e + (p_e - p_a) * A_e;
T_g_full_throttle = T_e_with_losses * n_eg_g;
T_ng_full_throttle = T_e_with_losses * n_eg_ng;
I_z = 2e10;    % From 2e10 -> 2e8

v_ex = 3433.5;
m_dot_e = T_e_with_losses/v_ex;

theta_0 = deg2rad(90);

A_pitch_rate = [0 0; 1 0];
B_pitch_rate = [(d_tcg * T_g_full_throttle / I_z); 0];
C_pitch_rate = [1 0];
D_pitch_rate = 0;
IC_pitch_ss = [0 theta_0];

C_pitch = [0, 1];
D_pitch = 0;

sys = ss(A_pitch_rate, B_pitch_rate, C_pitch_rate, D_pitch_rate, 0.01);

%% Pitch reference
dt = 0.01;
T_final = 120;
t = 0:dt:T_final;
T = T_final; % For convenience

% Pitch reference coefficients
a0 = 90;
a1 = 0;
a3 = 90 / (T^3);
a2 = -1.5 * a3 * T;

% Compute the pitch reference (in degrees)
theta_ref = a0 + a1 * t + a2 * t.^2 + a3 * t.^3;

% Compute pitch rate using finite differences (deg/s)
pitch_rate = diff(theta_ref) / dt;

% Find the maximum absolute pitch rate step and its index
[max_rate, idx] = max(abs(pitch_rate));

% Display the results
fprintf('Maximum pitch rate step: %.4f deg/s at index %d (t = %.2f s)\n', ...
    max_rate, idx, t(idx));

max_rate_rad = deg2rad(max_rate);

%% Tuned pitch rate gains
Kp_pitch_rate = 37.2;
Ki_pitch_rate = 5.48;
Kd_pitch_rate = 27.0606;
N_pitch_rate = 18.2;

%%
dt_sim = 0.1;