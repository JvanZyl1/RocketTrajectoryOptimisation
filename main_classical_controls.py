from src.classical_controls.ascent_control import AscentControl
from src.classical_controls.flip_over_and_boostbackburn_control import FlipOverandBoostbackBurnControl
from src.classical_controls.ballisitic_arc_descent import HighAltitudeBallisticArcDescent, BallisticArcDescentTuning
from src.classical_controls.re_entry_burn import ReEntryBurn
from src.classical_controls.landing_burn_optimise import LandingBurnOptimiser

#ascent_control = AscentControl()
#ascent_control.run_closed_loop()

#flip_over_and_boostbackburn_control = FlipOverandBoostbackBurnControl(pitch_tuning_bool = False)
#flip_over_and_boostbackburn_control.run_closed_loop()


# Uncomment to run the tuning for ballistic arc descent
ballistic_arc_tuning = BallisticArcDescentTuning(tune_bool=True)
ballistic_arc_tuning.run_closed_loop()

# Run the standard ballistic arc controller
ballistic_arc_descent = HighAltitudeBallisticArcDescent()
ballistic_arc_descent.run_closed_loop()

# Uncomment to run the tuning for re-entry burn
#re_entry_burn = ReEntryBurn(tune_bool=True)
#re_entry_burn.run_closed_loop()

# Run the standard re-entry burn controller
#re_entry_burn = ReEntryBurn(tune_bool=False)
#re_entry_burn.run_closed_loop()

#landing_burn_optimiser = LandingBurnOptimiser()
#landing_burn_optimiser()



