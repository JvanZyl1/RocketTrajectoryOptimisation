

class ReEntryBurn:
    def __init__(self):
        self.dt = 0.1

        self.simulation_step_lambda = compile_physics(dt = self.dt,
                                                      flight_phase = 're_entry_burn')
        self.state = load_re_entry_burn_initial_state('supervisory')
        self.initialise_logging()
        self.initial_condition()

    def initialise_logging(self):
        self.x_vals = []
        self.y_vals = []
        self.pitch_angle_deg_vals = []
        self.pitch_angle_reference_deg_vals = []
        self.time_vals = []
        self.flight_path_angle_deg_vals = []
        self.mach_number_vals = []
        self.angle_of_attack_deg_vals = []
        self.pitch_rate_deg_vals = []
        self.mass_vals = []
        self.mass_propellant_vals = []
        self.vx_vals = []
        self.vy_vals = []

    def initial_condition(self):
        pass

    def reset(self):
        pass

    def closed_loop_step(self):
        pass

    def save_results(self):
        pass

    def plot_results(self):
        pass

    def run_closed_loop(self):
        pass  