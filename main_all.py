from src.RocketSizing.main_sizing import size_rocket
from src.classical_controls.ascent_control import AscentControl
from src.supervisory_learning.supervisory_learn import SupervisoryLearning
from src.classical_controls.flip_over_and_boostbackburn_control import FlipOverandBoostbackBurnControl
from src.classical_controls.ballisitic_arc_descent import HighAltitudeBallisticArcDescent, BallisticArcDescentTuning
from src.classical_controls.landing_burn import LandingBurn, tune_landing_burn, reference_landing_trajectory
from src.trainers.trainer_rocket_SAC import RocketTrainer_ReinforcementLearning

# Run the rocket sizing
dv_loss_a_1 = 850.0
dv_loss_a_2 = 50.0
dv_loss_d_1 = 1800.0
eps_d_1 = 0.6
size_rocket(dv_loss_a_1, dv_loss_a_2, dv_loss_d_1, eps_d_1)

# Run the ascent control
ascent_control = AscentControl()
ascent_control.run_closed_loop()

# Run the subsonic supervisory learning
subsonic_supervisory = SupervisoryLearning(flight_phase='subsonic')
subsonic_supervisory()
supersonic_supervisory = SupervisoryLearning(flight_phase='supersonic')
supersonic_supervisory()

# Run the tuned flip_over_and_boostbackburn control
flip_over_and_boostbackburn_control = FlipOverandBoostbackBurnControl(pitch_tuning_bool=False)
flip_over_and_boostbackburn_control.run_closed_loop()

# Run the flip_over_and_boostbackburn supervisory learning
flip_over_and_boostbackburn_supervisory = SupervisoryLearning(flight_phase='flip_over_boostbackburn')
flip_over_and_boostbackburn_supervisory()

# Run the ballistic arc descent tuning
ballistic_arc_tuning = BallisticArcDescentTuning(tune_bool=True)
ballistic_arc_tuning.run_closed_loop()

# Run the ballistic arc descent control
ballistic_arc_descent = HighAltitudeBallisticArcDescent()
ballistic_arc_descent.run_closed_loop()

# Run the ballistic arc descent supervisory learning
ballistic_arc_supervisory = SupervisoryLearning(flight_phase='ballistic_arc_descent')
ballistic_arc_supervisory()

# Run the landing burn reference trajectory
ref = reference_landing_trajectory()

# Run the landing burn tuning
tune_landing_burn()

# Run the landing burn control
landing_burn = LandingBurn()
landing_burn.run_closed_loop()

# Run the landing burn supervisory learning
landing_burn_supervisory = SupervisoryLearning(flight_phase='landing_burn')
landing_burn_supervisory()

# Run the landing burn reinforcement learning
trainer = RocketTrainer_ReinforcementLearning(flight_phase = 'landing_burn',
                             load_from = 'supervisory',
                             load_buffer_bool= False,
                             save_interval = 5,
                             pre_train_critic_bool = False,
                             buffer_type = 'priotised',
                             rl_type = 'td3',
                             enable_wind = True)
trainer()

