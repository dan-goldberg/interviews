import numpy as np

def get_launch_trajectory_vector(horizonal_angle):
    """[summary]

    Args:
        horizonal_angle ([type]): [description]

    Returns:
        [type]: [description]
    """
    hypotenuse = 250
    y = hypotenuse * np.cos(degrees_to_radians(horizonal_angle))
    x = hypotenuse * np.sin(degrees_to_radians(horizonal_angle))
    return x, y

def get_orthogonal_projection_vector(vector, projection_surface):
    """[summary]

    Args:
        vector ([type]): [description]
        projection_surface ([type]): [description]

    Returns:
        [type]: [description]
    """
    # https://en.wikibooks.org/wiki/Linear_Algebra/Orthogonal_Projection_Onto_a_Line
    return np.dot((np.dot(vector, projection_surface) / np.dot(projection_surface, projection_surface)), projection_surface)

def degrees_to_radians(degrees):
    return degrees * 2 * np.pi / 360

def radians_to_degrees(radians):
    return (radians * 360) / (2 * np.pi)

def likely_interception_point(
    launch_speed, 
    horizonal_angle, 
    player_x, 
    player_y,
    player_speed=21.0
    ):
    """[summary]

    Args:
        launch_speed ([type]): [description]
        horizonal_angle ([type]): [description]
        player_x ([type]): [description]
        player_y ([type]): [description]
        player_speed (int or float, optional): Feet per second. Default to 20.0.

    Returns:
        [type]: [description]
    """
    # need to find the vector paralell with trajectory vector 
    # which minimizes the ball flight time given the constraints
    # player movement time = ball flight time

    player_speed = player_speed # feet per second
    ball_speed = mph_to_feet_per_second(launch_speed*0.95)
        # this should probably be a more complicated 
        # function of launch speed and how close to home the ball landed
    launch_trajectory = get_launch_trajectory_vector(horizonal_angle)
    m = launch_trajectory[1]/launch_trajectory[0] # y/x to get m from y=mx

    # coefficients of quadratic equation ax^2 + bx + c = 0
    a = (1 - (player_speed / ball_speed)**2) * (m**2 + 1)
    b = -2 * (player_x + ( m * player_y))
    c = (player_x**2) + (player_y**2)

    # solve quadratic formula
    roots = np.roots([a, b, c]) 
    roots_is_real = ~np.iscomplex(roots)
    if not roots_is_real.any():
        return None # if there are no real roots, the player cannot intercept the ball given assumptions
    else:
        roots = roots[roots_is_real] # only look at the real root(s)
        preferred_root_index = np.argmin(roots**2) # take the root closer to home plate
        min_time_x = roots[preferred_root_index] 
        min_time_y = min_time_x*m
        return np.array([min_time_x, min_time_y])

def get_angle_between_vectors(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return radians_to_degrees(angle)

def player_approach_angle(player_position, throw_position, base_position):
    return get_angle_between_vectors(
        throw_position - player_position, 
        base_position - player_position
        )

def distance_between_coords(vector_1, vector_2):
    return np.linalg.norm(vector_1 - vector_2)

def mph_to_feet_per_second(mph):
    # feet_per_mile = 5280
    return (mph * 5280) / (60**2)
