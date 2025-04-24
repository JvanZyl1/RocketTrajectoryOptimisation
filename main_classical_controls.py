from src.classical_controls.ascent_control import AscentControl
from src.classical_controls.flip_over_and_boostbackburn_control import FlipOverandBoostbackBurnControl
from src.classical_controls.ballisitic_arc_descent import HighAltitudeBallisticArcDescent
from src.classical_controls.re_entry_burn import ReEntryBurn
from src.classical_controls.landing_burn_optimise import LandingBurnOptimiser
'''
ascent_control = AscentControl()
ascent_control.run_closed_loop()

flip_over_and_boostbackburn_control = FlipOverandBoostbackBurnControl(pitch_tuning_bool = False)
flip_over_and_boostbackburn_control.run_closed_loop()

ballistic_arc_descent = HighAltitudeBallisticArcDescent()
ballistic_arc_descent.run_closed_loop()

re_entry_burn = ReEntryBurn(tune_ACS_bool = False)
re_entry_burn.run_closed_loop()

landing_burn_optimiser = LandingBurnOptimiser()
landing_burn_optimiser()


'''



