import math

def new_radius_func(number_of_engines_new : int):
    # For super heavy booster
    diameter_engine = 1.3
    rocket_diameter = 9
    number_of_inner_engines = 3
    number_of_middle_engines = 10
    number_of_outer_engines = 20
    number_of_engines_super_heavy = 33
    clearance_outer_engines = 2 * math.pi/number_of_outer_engines * (rocket_diameter / 2 + diameter_engine / 2) - diameter_engine
    rocket_diameter_check = (clearance_outer_engines + diameter_engine) * number_of_outer_engines / ( math.pi) - diameter_engine

    # use is close to check the diameter
    assert math.isclose(rocket_diameter, rocket_diameter_check, rel_tol=1e-3), f'Rocket diameter is not correct: {rocket_diameter} vs {rocket_diameter_check}'

    # New rocket circumference
    number_of_engines_added = number_of_engines_new - number_of_engines_super_heavy
    if number_of_engines_added == 0:
        number_of_outer_engines_added = 0
        number_of_middle_engines_added = 0
    elif number_of_engines_added < 0: # Removed engines, in order: middle, outer, outer, middle, outer, outer, ...
        number_of_middle_engines_added = 0
        number_of_outer_engines_added = 0
        for i in range(abs(number_of_engines_added)):
            if i % 3 == 0:
                number_of_middle_engines_added -= 1
            else:
                number_of_outer_engines_added -= 1
    else: # Added engines, in order: outer, outer, middle, outer, outer, middle, ...
        number_of_middle_engines_added = 0
        number_of_outer_engines_added = 0
        for i in range(number_of_engines_added):
            if i % 3 == 0:
                number_of_middle_engines_added += 1
            else:
                number_of_outer_engines_added += 1

    number_of_engines_per_ring = [number_of_inner_engines,
                                  number_of_middle_engines + number_of_middle_engines_added,
                                  number_of_outer_engines + number_of_outer_engines_added]

    number_of_outer_engines_new = number_of_engines_per_ring[2]

    # THIS IS WRONG FIX IT.
    if number_of_outer_engines_new % 2 == 0:
        number_of_engines_even = number_of_outer_engines_new
    else:
        number_of_engines_even = number_of_outer_engines_new + 1 # Just for the space

    print(f'Number of engines even: {number_of_engines_even}')
    print(f'Clearance outer engines: {clearance_outer_engines}')
    print(f'Diameter engine: {diameter_engine}')
    rocket_diameter_new = (clearance_outer_engines + diameter_engine) * number_of_engines_even / ( math.pi) - diameter_engine
    rocket_radius_new = rocket_diameter_new / 2

    frontal_area_new = math.pi * (rocket_radius_new ** 2)
    return rocket_radius_new, frontal_area_new, number_of_engines_per_ring