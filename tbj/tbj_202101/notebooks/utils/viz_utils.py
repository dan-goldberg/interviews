import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg # https://matplotlib.org/3.3.3/tutorials/introductory/images.html
# https://image.online-convert.com/convert/svg-to-png
from PIL import Image

from . import geometry

class Diamond:
    
    def __init__(self, 
                 figsize=(8,8), 
                 scatter_kwargs=dict(s=10, alpha=1.0)
                ):
        """
        A class that exposes a to-scale plot of a pro diamond and
        helper function to plot scatter points with units in feet from home plate.
        
        All input units are feet with the plane centred on home plate. 
        
        You can scatter plot coordinates and also line segments.
        """
        # base image details
        self.img = Image.open('../assets/BaseballDiamondScale.jpg')    # Open image as PIL image object
        self.HOME_X = 647
        self.HOME_Y = 1061
        self.SECOND_BASE_Y = 767
        
        # constants for plotting coordinates
        FEET_BETWEEN_HOME_AND_SECOND = np.math.pow(2 * np.math.pow(90,2), 1/2) 
        self.UNITS_PER_FEET = (self.HOME_Y - self.SECOND_BASE_Y) / FEET_BETWEEN_HOME_AND_SECOND # 127.279 feet
        
        # coordinates with attributes
        self.scatter_kwargs = scatter_kwargs
        self.coords = [] #empty list of coordinates
        
        # for line segment
        self.lines_x = []
        self.lines_y = []

        # base positions in feet

        self.HOME_VECTOR = np.array([0, 0])

        self.FIRST_BASE_VECTOR = self._calc_vector_in_feet(
            *np.array([
                self.HOME_X + (self.HOME_Y-self.SECOND_BASE_Y)/2,
                self.HOME_Y - (self.HOME_Y-self.SECOND_BASE_Y)/2
            ])
        )

        self.SECOND_BASE_VECTOR = (0, FEET_BETWEEN_HOME_AND_SECOND)

        self.THIRD_BASE_VECTOR = self._calc_vector_in_feet(
            *np.array([
                self.HOME_X - (self.HOME_Y-self.SECOND_BASE_Y)/2,
                self.HOME_Y - (self.HOME_Y-self.SECOND_BASE_Y)/2
            ])
        )
        
    def set_coord(self, x, y, **kwargs):
        self.coords.append((self._calc_coord(x,y), dict(**kwargs)))
        
    def _calc_coord(self, x, y):
        # x and y in feet from home
        return (x*self.UNITS_PER_FEET)+self.HOME_X, ((-y*self.UNITS_PER_FEET)+self.HOME_Y)
        
    def _calc_vector_in_feet(self, x, y):
        # x and y in coords
        return (x - self.HOME_X) / self.UNITS_PER_FEET, (y - self.HOME_Y) / (-self.UNITS_PER_FEET)
    
    def plot_line_segment(self, coord1, coord2):
        reshaped = list(zip(self._calc_coord(*coord1), self._calc_coord(*coord2)))
        self.lines_x.append(list(reshaped[0]))
        self.lines_y.append(list(reshaped[1]))
        
    def show(self, title=None):
        fig = plt.figure(figsize=(9,9))
        plt.axis('off')
        plt.imshow(self.img)
        coords = np.array(self.coords)
        for line_x, line_y in zip(self.lines_x, self.lines_y):
            plt.plot(line_x, line_y)
        if self.coords:
            for coord in self.coords:
                plt.scatter(coord[0][0], coord[0][1], **dict(**coord[1], **self.scatter_kwargs))
        if title:
            plt.title(title)
        plt.legend()
        fig.show()
        
        
def plot_single_sample(sample):

    traj = geometry.get_launch_trajectory_vector(sample['launch_horiz_ang'].values[0])
    position = sample[['player_x', 'player_y']].values[0]
    projection = geometry.get_orthogonal_projection_vector(position, traj)
    landing = sample[['landing_location_x', 'landing_location_y']].values[0]
    base = sample['base_of_interest_point'].values[0]

    likely_interception_point = geometry.likely_interception_point(
        launch_speed = sample['launch_speed'].values[0],
        horizonal_angle = sample['launch_horiz_ang'].values,
        player_x = sample['player_x'].values[0],
        player_y = sample['player_y'].values[0]
    )

    diamond = Diamond(scatter_kwargs={'s': 30})
    diamond.plot_line_segment((0,0), traj)
    diamond.plot_line_segment(position, projection)
    diamond.set_coord(*position, label='player')
    diamond.set_coord(*landing, label='landing')
    diamond.set_coord(*base, label='base of interest')
    if likely_interception_point is not None:
        diamond.set_coord(*likely_interception_point, label='inferred intersection')
    diamond.show(title=f"""
    {sample['id'].iloc[0]}
    {sample['playtype'].iloc[0]} | {sample['eventtype'].iloc[0]} | {'-'.join(sample['fieldingplay'].iloc[0])} | {sample['fielded_scoring'].iloc[0]}
    launch_speed: {sample['launch_speed'].iloc[0]:.1f} | launch_vert_ang: {sample['launch_vert_ang'].iloc[0]:.1f}
    base_of_interest: {sample['base_of_interest'].iloc[0]} | angle: {sample['player_angle_from_interception_point_to_base_of_interest'].iloc[0]:.1f}
    player_time: {sample['player_time_to_point_of_interest'].iloc[0]:.2f} | ball_time: {sample['ball_time_to_point_of_interest'].iloc[0]:.2f}""")



if __name__ == '__main__':

    fig = plt.figure(figsize=(15,15))
    img = Image.open('../../assets/BaseballDiamondScale.jpg')    # Open image as PIL image object

    imgplot = plt.imshow(img)
    #plt.axis('off')

    plt.scatter([647], [1061], s=50) #found home plate by aligning this coordinate with the plate
    plt.scatter([647], [767], s=50) #found 2b by aligning this coordinate with the base
    distance_from_home_to_second = 1061 - 767
    plt.scatter([647+distance_from_home_to_second/2], [1061-distance_from_home_to_second/2], s=50) #found 1b by interpolation
    plt.scatter([647-distance_from_home_to_second/2], [1061-distance_from_home_to_second/2], s=50) #found 3b by interpolation
    # interpolated points on 1b and 3b confirms infield is to scale
    plt.show()
