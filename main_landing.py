#!/usr/bin/env python3
"""
Rocket landing – trajectory generation and basic post-processing.

Units: m, s, rad.  British English spelling.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plan_trajectory(x0, y0, vx0, vy0, pitch0,
                    dt=0.1, max_pitch_rate=np.deg2rad(5),
                    a_lat_max=5.0, a_total_max=30.0, g=9.81):
    t, x, y, vx, vy, pitch = [0.0], [x0], [y0], [vx0], [vy0], [pitch0]
    while y[-1] > 0.0 and t[-1] < 300.0:
        c_x, c_y, c_vx, c_vy, c_th = x[-1], y[-1], vx[-1], vy[-1], pitch[-1]

        # Required horizontal deceleration
        a_h_req = (c_vx ** 2) / (2 * abs(c_x)) if abs(c_x) > 1e-3 and abs(c_vx) > 1e-3 else 0.0
        a_h_req = min(a_h_req, a_lat_max)
        a_h = -np.sign(c_vx) * a_h_req if abs(c_vx) > 1e-3 else -np.sign(c_x) * a_h_req

        # Required vertical thrust component
        net_up = (c_vy ** 2) / (2 * c_y) if c_y > 0.0 and c_vy < 0.0 else 0.0
        a_v = net_up + g

        # Desired pitch limited by actuator rate
        th_des = np.arctan2(abs(a_h), a_v) * np.sign(a_h)
        th_des = np.clip(th_des, -np.deg2rad(85), np.deg2rad(85))
        dth_max = max_pitch_rate * dt
        n_th = np.clip(th_des, c_th - dth_max, c_th + dth_max)

        # Thrust magnitude subject to limits
        thrust = a_v / max(np.cos(n_th), 1e-6)
        if abs(thrust * np.sin(n_th)) > a_lat_max:
            thrust = a_lat_max / (abs(np.sin(n_th)) + 1e-6)
        thrust = min(thrust, a_total_max)

        ax, az = thrust * np.sin(n_th), thrust * np.cos(n_th) - g

        # Integrate
        nvx, nvy = c_vx + ax * dt, c_vy + az * dt
        nx, ny = c_x + c_vx * dt + 0.5 * ax * dt ** 2, c_y + c_vy * dt + 0.5 * az * dt ** 2

        if ny <= 0.0:  # touchdown interpolation
            tau = (-c_y) / (c_vy + 1e-10) if abs(c_vy) > 1e-3 else 0.0
            tau = np.clip(tau, 0.0, dt)
            t.append(t[-1] + tau)
            x.append(c_x + c_vx * tau + 0.5 * ax * tau ** 2)
            y.append(0.0)
            vx.append(c_vx + ax * tau)
            vy.append(c_vy + az * tau)
            pitch.append(n_th)
            break

        t.append(t[-1] + dt), x.append(nx), y.append(ny)
        vx.append(nvx), vy.append(nvy), pitch.append(n_th)

    return (np.asarray(t), np.asarray(x), np.asarray(y),
            np.asarray(vx), np.asarray(vy), np.asarray(pitch))


def save_csv(fname, t, x, y, vx, vy, th):
    hdr = ["t", "x", "y", "vx", "vy", "theta"]
    with Path(fname).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(zip(t, x, y, vx, vy, th))


def summary(t, x, y, vx, vy, th):
    ax = np.gradient(vx, t)
    a_lat = np.abs(ax)
    print("Flight time, s:      ", f"{t[-1]:.1f}")
    print("Touch-down vx, m/s:  ", f"{vx[-1]:.2f}")
    print("Touch-down vy, m/s:  ", f"{vy[-1]:.2f}")
    print("Peak lateral g:      ", f"{np.max(a_lat) / 9.81:.2f}")
    print("Peak pitch, deg:     ", f"{np.rad2deg(np.max(np.abs(th))):.1f}")


def single_plot(x, y, xlabel, ylabel, title):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()


def main():
    # Initial state: horizontal 2 km, altitude 1.5 km, closing at 80 m/s, descending at 50 m/s
    t, x, y, vx, vy, th = plan_trajectory(2000.0, 1500.0, -80.0, -50.0, -np.deg2rad(20))

    save_csv("trajectory.csv", t, x, y, vx, vy, th)
    summary(t, x, y, vx, vy, th)

    single_plot(x, y, "Horizontal distance, m", "Altitude, m", "Down-range profile")
    single_plot(t, vx, "Time, s", "vx, m/s", "Horizontal velocity history")
    single_plot(t, vy, "Time, s", "vy, m/s", "Vertical velocity history")
    single_plot(t, np.rad2deg(th), "Time, s", "Pitch, deg", "Pitch angle history")

    plt.show()


if __name__ == "__main__":
    main()
