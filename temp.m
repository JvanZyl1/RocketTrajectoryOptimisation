%% PID tuning for RCS rotational axis

clear all
close all
clc

%% Constants
x_cog = 47.124085480011864;                    % m
inertia = 11172563283.216793;                  % kg·m²
RCS_thruster_per_group = 50000;                % N
d_base_rcs_top = 103.192;                      % m

Moment_max = RCS_thruster_per_group * (1 + d_base_rcs_top - 2 * x_cog); % N·m

%% Plant model (rigid-body rotation)
s = tf('s');
I = inertia;
G = 1/(I*s^2);

desiredRiseTime = 1;           % s
BW = 1.8/desiredRiseTime;      % bandwidth approximation

%% Initial PID design
[C, info] = pidtune(G, 'PID', BW);

%% Gain scaling to keep |y| ≤ 1
scale = 1;
sat = 1;
t = 0:0.1:40;

while true
    Cscaled = scale * C;
    T = feedback(Cscaled * G, 1);
    y = step(T, t);
    if max(abs(y)) <= sat
        break
    end
    scale = 0.9 * scale;      % reduce gain
end

fprintf("Final gains: Kp = %.6g, Ki = %.6g, Kd = %.6g\n", Cscaled.Kp, Cscaled.Ki, Cscaled.Kd);

%% Step response plot and save
figure
step(T, t)
grid on
title('Closed-loop step response')
xlabel('Time (s)')
ylabel('Output')
ylim([-1.2 1.2])
exportgraphics(gcf, 'step_response.png')   % save figure

%% Pole-zero map and save
figure
pzmap(T)
grid on
title('Closed-loop poles and zeros')
exportgraphics(gcf, 'pole_zero_map.png')   % save figure
