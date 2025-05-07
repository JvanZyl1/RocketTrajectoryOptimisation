import csv
from src.RocketSizing.main_sizing import size_rocket
from src.classical_controls.ascent_control import AscentControl
from src.supervisory_learning.supervisory_learn import SupervisoryLearning
from src.classical_controls.flip_over_and_boostbackburn_control import FlipOverandBoostbackBurnControl
from src.classical_controls.ballisitic_arc_descent import HighAltitudeBallisticArcDescent, BallisticArcDescentTuning
from src.particle_swarm_optimisation.particle_swarm_optimisation import ParticleSubswarmOptimisation

# Run the rocket sizing
converged = False
dv_loss_a_1 = 120.0
dv_loss_a_2 = 710.0
dv_loss_d_1 = 0.0
eps_d_1 = 0.28
iter_count = 1
while converged == False:
    size_rocket(dv_loss_a_1, dv_loss_a_2, dv_loss_d_1, eps_d_1)
    # Run the ascent control
    ascent_control = AscentControl()
    ascent_control.reset()
    ascent_control.run_closed_loop()
    # Check if the ascent control has converged
    with open('data/rocket_parameters/velocity_increments.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == '(controller results) delta_v_a_1_loss_error':
                delta_v_a_1_loss_error = float(row[1])
            if row[0] == '(controller results) delta_v_a_1_loss':
                dv_loss_a_1 = float(row[1])
    print(f'Iteration {iter_count}, delta_v_a_1_loss_error: {delta_v_a_1_loss_error:.2f} m/s')
    if abs(delta_v_a_1_loss_error) < 10:
        converged = True
    iter_count += 1

# Run the subsonic supervisory learning
subsonic_supervisory = SupervisoryLearning(flight_phase='subsonic')
subsonic_supervisory()
supersonic_supervisory = SupervisoryLearning(flight_phase='supersonic')
supersonic_supervisory()

flip_over_and_boostbackburn_control = FlipOverandBoostbackBurnControl(pitch_tuning_bool = False)
flip_over_and_boostbackburn_control.run_closed_loop()

flip_over_and_boostbackburn_supervisory = SupervisoryLearning(flight_phase='flip_over_boostbackburn')
flip_over_and_boostbackburn_supervisory()

ballistic_arc_tuning = BallisticArcDescentTuning(tune_bool=True)
ballistic_arc_tuning.run_closed_loop()
ballistic_arc_descent = HighAltitudeBallisticArcDescent()
ballistic_arc_descent.run_closed_loop()

ballistic_arc_supervisory = SupervisoryLearning(flight_phase='ballistic_arc_descent')
ballistic_arc_supervisory()

re_entry_burn_pso = ParticleSubswarmOptimisation(flight_phase= 're_entry_burn', save_interval = 5, enable_wind = False)
re_entry_burn_pso()
