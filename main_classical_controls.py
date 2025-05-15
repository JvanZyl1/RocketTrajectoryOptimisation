from src.classical_controls.ascent_control import AscentControl
from src.classical_controls.flip_over_and_boostbackburn_control import FlipOverandBoostbackBurnControl
from src.classical_controls.ballisitic_arc_descent import HighAltitudeBallisticArcDescent, BallisticArcDescentTuning
from src.classical_controls.landing_burn import LandingBurn, tune_landing_burn, reference_landing_trajectory

#ascent_control = AscentControl()
#ascent_control.run_closed_loop()

#flip_over_and_boostbackburn_control = FlipOverandBoostbackBurnControl(pitch_tuning_bool = False)
#flip_over_and_boostbackburn_control.run_closed_loop()

#ballistic_arc_tuning = BallisticArcDescentTuning(tune_bool=True)
#ballistic_arc_tuning.run_closed_loop()

#ballistic_arc_descent = HighAltitudeBallisticArcDescent()
#ballistic_arc_descent.run_closed_loop()

#ref = reference_landing_trajectory()
#tune_landing_burn()
landing_burn = LandingBurn()
landing_burn.run_closed_loop()



