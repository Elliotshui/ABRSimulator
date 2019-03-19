import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self, x, y, title):
        self.x = x
        self.y = y
        self.title = title
        return 

    def plot_info(self):
        fig = plt.figure()
        subplt = fig.add_subplot(111)
        subplt.plot(self.x, self.y)
        subplt.set_title(self.title)
        return fig