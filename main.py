from src.envs.env_endo.main_env_endo import rocket_model_endo_ascent

env = rocket_model_endo_ascent(sizing_needed_bool = False)

env.run_test_physics()