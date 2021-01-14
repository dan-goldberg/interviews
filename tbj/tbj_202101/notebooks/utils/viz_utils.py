import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg # https://matplotlib.org/3.3.3/tutorials/introductory/images.html
# https://image.online-convert.com/convert/svg-to-png
from PIL import Image

class Diamond:
    
    def __init__(self, 
                 figsize=(8,8), 
                 scatter_kwargs=dict(s=10, c=None, alpha=1.0)
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
        
    def set_coord(self, x, y):
        self.coords.append(self._calc_coord(x,y))
        
    def _calc_coord(self, x, y):
        # x and y in feet from home
        return (x*self.UNITS_PER_FEET)+self.HOME_X, ((-y*self.UNITS_PER_FEET)+self.HOME_Y)
    
    def plot_line_segment(self, coord1, coord2):
        reshaped = list(zip(self._calc_coord(*coord1), self._calc_coord(*coord2)))
        self.lines_x.append(list(reshaped[0]))
        self.lines_y.append(list(reshaped[1]))
        
    def show(self):
        fig = plt.figure(figsize=(9,9))
        plt.axis('off')
        plt.imshow(self.img)
        coords = np.array(self.coords)
        for line_x, line_y in zip(self.lines_x, self.lines_y):
            plt.plot(line_x, line_y)
        if self.coords:
            plt.scatter(coords[:,0], coords[:,1], **self.scatter_kwargs)
        fig.show()


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
