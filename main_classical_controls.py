from src.classical_controls.ascent_control import AscentControl
from src.classical_controls.flip_over_control import FlipOverControl
ascent_control = AscentControl()
ascent_control.run_closed_loop()

flip_over_control = FlipOverControl()
flip_over_control.run_closed_loop()