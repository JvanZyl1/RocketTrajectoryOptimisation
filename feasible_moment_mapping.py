import pandas as pd
df = pd.read_csv('data/reference_trajectory/ballistic_arc_descent_controls/state_action_ballistic_arc_descent_control.csv')
max_propellant_mass = df['mass_propellant[kg]'].max()
