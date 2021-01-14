import numpy as np

def get_launch_trajectory_vector(horizonal_angle):
    hypotenuse = 150
    y = hypotenuse * np.cos(degrees_to_radians(horizonal_angle))[0]
    x = hypotenuse * np.sin(degrees_to_radians(horizonal_angle))[0]
    return x, y

def get_orthogonal_projection_vector(vector, projection_surface):
    # https://en.wikibooks.org/wiki/Linear_Algebra/Orthogonal_Projection_Onto_a_Line
    return np.dot((np.dot(vector, projection_surface) / np.dot(projection_surface, projection_surface)), projection_surface)

def degrees_to_radians(degrees):
    return degrees * 2 * np.pi / 360