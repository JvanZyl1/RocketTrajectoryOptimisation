import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.envs.utils.aerodynamic_coefficients import rocket_CD_compiler, rocket_CL_compiler
from src.envs.load_initial_states import load_landing_burn_initial_state
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model

class AerodynamicStabilityDescent:
    def __init__(self):
        self.state = load_landing_burn_initial_state()
        self.x, self.y, self.vx, self.vy, self.theta, self.theta_dot, self.gamma, self.alpha, self.mass, self.mass_propellant, self.time = self.state
        self.CD_func = lambda M, alpha_rad: rocket_CD_compiler()(M, math.degrees(alpha_rad)) # Mach, alpha [deg]
        self.CL_func = lambda M, alpha_rad: rocket_CL_compiler()(M, math.degrees(alpha_rad)) # Mach, alpha [deg]
        # Read sizing results
        sizing_results = {}
        with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sizing_results[row[0]] = row[2]
        self.m_burn_out = float(sizing_results['Structural mass stage 1 (ascent)'])*1000
        self.T_e = float(sizing_results['Thrust engine stage 1'])
        self.v_ex = float(sizing_results['Exhaust velocity stage 1'])
        self.n_e = 6 # select a realistic number of engines
        self.mdot = self.T_e/self.v_ex * self.n_e
        self.frontal_area = float(sizing_results['Rocket frontal area'])
        self.x_cog = 20
        self.x_cop = 0.25 * float(sizing_results['Stage 1 height '])

        self.dt = 0.1
        self.inertia = 1e9
        self.alpha_effective = self.gamma - self.theta - math.pi
        self.initialise_logging()

    def initialise_logging(self):
        self.x_log = []
        self.y_log = []
        self.vx_log = []
        self.vy_log = []
        self.theta_log = []
        self.theta_dot_log = []
        self.gamma_log = []
        self.alpha_effective_log = []
        self.aero_x_log = []
        self.aero_y_log = []
        self.aero_moments_z_log = []
        self.lift_log = []
        self.drag_log = []
        self.alpha_log = []
        self.mass_log = []
        self.mass_propellant_log = []
        self.time_log = []
        self.mach_number_log = []
        self.dynamic_pressure_log = []
        self.CL_log = []
        self.CD_log = []

    def log_data(self):
        self.x_log.append(self.x)
        self.y_log.append(self.y)
        self.vx_log.append(self.vx)
        self.vy_log.append(self.vy)
        self.theta_log.append(self.theta)
        self.theta_dot_log.append(self.theta_dot)
        self.gamma_log.append(self.gamma)
        self.alpha_effective_log.append(self.alpha_effective)
        self.aero_x_log.append(self.aero_x)
        self.aero_y_log.append(self.aero_y)
        self.aero_moments_z_log.append(self.aero_moments_z)
        self.lift_log.append(self.lift)
        self.drag_log.append(self.drag)
        self.alpha_log.append(self.alpha)
        self.mass_log.append(self.mass)
        self.mass_propellant_log.append(self.mass_propellant)
        self.time_log.append(self.time)
        self.dynamic_pressure_log.append(self.dynamic_pressure)    
        self.CL_log.append(self.C_L)
        self.CD_log.append(self.C_D)  
        self.mach_number_log.append(self.mach_number)  

    def step(self):
        self.alpha_effective = (self.theta + math.pi) - self.gamma
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(self.y)
        speed = math.sqrt(self.vx**2 + self.vy**2)
        self.dynamic_pressure = 0.5 * density * speed**2
        if speed_of_sound != 0.0:
            self.mach_number = speed/speed_of_sound
            self.C_L = self.CL_func(self.mach_number, self.alpha_effective) # Mach, alpha [rad]
            self.C_D = self.CD_func(self.mach_number, self.alpha_effective) # Mach, alpha [rad]
        else:
            self.mach_number = 0.0
            self.C_L = 0.0
            self.C_D = 0.0
        self.drag = self.dynamic_pressure * self.C_D * self.frontal_area
        self.lift = self.dynamic_pressure * self.C_L * self.frontal_area

        aero_force_parallel = self.drag * math.cos(self.alpha_effective) + self.lift * math.sin(self.alpha_effective)
        aero_force_perpendicular = self.drag * math.sin(self.alpha_effective) - self.lift * math.cos(self.alpha_effective)
        d_cp_cg = self.x_cog - self.x_cop
        self.aero_x = aero_force_parallel * math.cos(self.theta) + aero_force_perpendicular * math.sin(self.theta)
        self.aero_y = aero_force_parallel * math.sin(self.theta) - aero_force_perpendicular * math.cos(self.theta)
        self.aero_moments_z = aero_force_perpendicular * d_cp_cg

        # Thrust
        T_parallel = self.T_e * self.n_e
        T_x = T_parallel * math.cos(self.theta)
        T_y = T_parallel * math.sin(self.theta)
        
        # Acceleration linear
        a_x = (self.aero_x + T_x)/self.mass
        a_y = (self.aero_y + T_y)/self.mass

        # Velocity update
        self.vx += a_x * self.dt
        self.vy += a_y * self.dt

        # Position update
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

        # Moments update
        theta_dot_dot = self.aero_moments_z / self.inertia
        self.theta_dot += theta_dot_dot * self.dt
        self.theta += self.theta_dot * self.dt
        self.gamma = math.atan2(self.vy, self.vx)

        if self.theta > 2 * math.pi:
            self.theta -= 2 * math.pi
        if self.gamma < 0:
            self.gamma = 2 * math.pi + self.gamma

        self.alpha = self.theta - self.gamma
        
        # Mass update
        self.mass -= self.mdot * self.dt
        self.mass_propellant -= self.mdot * self.dt

        # Time update
        self.time += self.dt

    def run_closed_loop(self):
        while self.mass_propellant > 0.0 and self.y > 0.0 and abs(self.alpha_effective) < math.radians(10):
            self.step()
            self.log_data()

    def plot_results(self):
        self.pitch_down_deg = np.rad2deg(np.array(self.theta_log)) + 180
        plt.figure(figsize=(20,15))
        plt.suptitle('Aerodynamic Stability Descent : No pitch damping', fontsize=16)
        # 3 x 3 plot
        # alpha effective   | Mach      | Dynamic Pressure
        # Lift              | Drag      | Moment
        # Pitch (down), FPA | Pitch Rate| CL and CD
        gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(self.time_log, np.rad2deg(np.array(self.alpha_effective_log)), linewidth=2, color='blue')
        ax1.set_ylabel(r'$\alpha_{eff}$ [$^{\circ}$]', fontsize=14)
        ax1.grid(True)
        
        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(self.time_log, np.array(self.mach_number_log), linewidth=2, color='blue')
        ax2.set_ylabel(r'$M$', fontsize=14)
        ax2.grid(True)

        ax3 = plt.subplot(gs[0, 2])
        ax3.plot(self.time_log, np.array(self.dynamic_pressure_log)/1e3, linewidth=2, color='blue')
        ax3.set_ylabel(r'$q$ [kPa]', fontsize=14)
        ax3.grid(True)

        ax4 = plt.subplot(gs[1, 0])
        ax4.plot(self.time_log, np.array(self.lift_log)/1e3, linewidth=2, color='blue')
        ax4.set_ylabel(r'$L$ [kN]', fontsize=14)
        ax4.grid(True)

        ax5 = plt.subplot(gs[1, 1])
        ax5.plot(self.time_log, np.array(self.drag_log)/1e3, linewidth=2, color='blue')
        ax5.set_ylabel(r'$D$ [kN]', fontsize=14)
        ax5.grid(True)

        ax6 = plt.subplot(gs[1, 2])
        ax6.plot(self.time_log, np.array(self.aero_moments_z_log)/1e3, linewidth=2, color='blue')
        ax6.set_ylabel(r'$M_z$ [kNm]', fontsize=14)
        ax6.grid(True)

        ax7 = plt.subplot(gs[2, 0])
        ax7.plot(self.time_log, np.array(self.pitch_down_deg), linewidth=2, color='blue', label='Pitch Down')
        ax7.plot(self.time_log, np.rad2deg(np.array(self.gamma_log)), linewidth=2, color='red', label='Flight Path Angle')
        ax7.set_ylabel(r'Angle [$^{\circ}$]', fontsize=14)
        ax7.set_xlabel(r'Time [s]', fontsize=14)
        ax7.grid(True)
        ax7.legend()

        ax8 = plt.subplot(gs[2, 1])
        ax8.plot(self.time_log, np.rad2deg(np.array(self.theta_dot_log)), linewidth=2, color='blue', label='Pitch Rate')
        ax8.set_ylabel(r'$\dot{\theta}$ [$^{\circ}$/s]', fontsize=14)
        ax8.set_xlabel(r'Time [s]', fontsize=14)
        ax8.grid(True)
        ax8.legend()

        ax9 = plt.subplot(gs[2, 2])
        ax9.plot(self.time_log, np.array(self.CL_log), linewidth=2, color='blue', label='CL')
        ax9.plot(self.time_log, np.array(self.CD_log), linewidth=2, color='red', label='CD')
        ax9.set_ylabel(r'Coefficient', fontsize=14)
        ax9.set_xlabel(r'Time [s]', fontsize=14)
        ax9.grid(True)
        ax9.legend()

        plt.show()


if __name__ == '__main__':
    test_aerodynamic_stability_descent = AerodynamicStabilityDescent()
    test_aerodynamic_stability_descent.run_closed_loop()
    test_aerodynamic_stability_descent.plot_results()

        
