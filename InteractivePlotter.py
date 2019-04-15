
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

class InteractiveFigure:
    '''
    TODO:
    BUGS
    side panels location: currently the side panels may cover the main axes labels.
    FEATURES
    interactive plotting: plot only whatever within the current limits.

    InteractiveFigure
    This class allows creation of 1-axes interactive figure, in which the limits
    of the axes can be modified interactively using pointer ("mouse") dragging or
    scrolling.
    It also supports printing information into a text-box next to the main axes.

    Limitations:
    No support in log-scaled axes.
    (Certainly there're additional unknown limitations)

    Credits:
    This class is heavily based on code examples from StackOverflow question:
    "Matplotlib plot zooming with scroll wheel"
    https://stackoverflow.com/questions/11551049/matplotlib-plot-zooming-with-scroll-wheel

    Written by Ido Greenberg, 2018
    '''

    def __init__(self, horizontal_panel=True, vertical_panel=True,
                 text_panel=False, text_fun=None):
        self.ax, self.axh, self.axv, self.axt = (None for _ in range(4))
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.xpress = None
        self.ypress = None
        self.xdrag = False
        self.ydrag = False
        self.drag_axes = None
        self.moved = None
        self.text_fun = None
        self.create_interactive_ax(horizontal_panel, vertical_panel,
                                   text_panel, text_fun)

    def create_interactive_ax(self, hpanel=True, vpanel=True, tpanel=True,
                              text_fun=None):
        # figure
        fig = plt.figure()
        gs = gridspec.GridSpec(70,100)
        # axes & panels
        x0 = 30 if tpanel else 0
        x1 = -6 if vpanel else 100
        dx0 = 11
        dx1 = 1
        y1 = 56 if hpanel else 70
        dy1 = 8
        if vpanel:
            self.axv = fig.add_subplot(gs[:y1, x1+dx1:], autoscale_on=False)
            self.axv.set_xticks(())
            self.axv.set_yticks(())
        if hpanel:
            self.axh = fig.add_subplot(gs[y1+dy1:, x0:x1], autoscale_on=False)
            self.axh.set_xticks(())
            self.axh.set_yticks(())
        if tpanel:
            self.axt = fig.add_subplot(gs[:, :x0-dx0], autoscale_on=False)
            self.axt.set_xticks(())
            self.axt.set_yticks(())
        self.text_fun = text_fun
        self.ax =  fig.add_subplot(gs[:y1, x0:x1], autoscale_on=False)
        self.sync_panels_scale()
        # interactivation
        self.make_interactive()

    def sync_panels_scale(self):
        if self.axh: self.axh.set_xlim(self.ax.get_xlim())
        if self.axv: self.axv.set_ylim(self.ax.get_ylim())

    def make_interactive(self):
        self.zoom_factory()
        self.pan_factory()

    def set_text_fun(self, fun):
        self.text_fun = fun

    def get_axes(self):
        return self.ax

    def zoom_factory(self, base_scale=1.5):
        def zoom(event):
            ax = event.inaxes
            if not ax in (self.ax, self.axh, self.axv): return
            # current plot range
            l,r = self.ax.get_xlim()
            d,u = self.ax.get_ylim()
            # pointer location
            x = event.xdata
            y = event.ydata
            # event classification
            if event.button == 'down':
                # zoom in
                s = base_scale
            elif event.button == 'up':
                # zoom out
                s = 1/base_scale
            else:
                raise ValueError("Invalid scroll_event.")
            # do zoom
            if ax == self.ax:
                self.ax.set_xlim(InteractiveFigure.zoom_range(l,r,x,s))
                self.ax.set_ylim(InteractiveFigure.zoom_range(d,u,y,s))
            elif ax == self.axh:
                self.ax.set_xlim(InteractiveFigure.zoom_range(l,r,x,s))
            elif ax == self.axv:
                self.ax.set_ylim(InteractiveFigure.zoom_range(d,u,y,s))
            self.sync_panels_scale()
            # draw
            ax.figure.canvas.draw()

        fig = self.ax.get_figure()
        fig.canvas.mpl_connect('scroll_event', zoom)

    def zoom_range(x1, x2, x, s):
        y1 = x - s*(x-x1)
        y2 = x + s*(x2-x)
        return y1,y2

    def pan_factory(self):
        def onPress(event):
            self.drag_axes = event.inaxes
            self.xdrag = self.drag_axes in (self.ax, self.axh)
            self.ydrag = self.drag_axes in (self.ax, self.axv)
            self.cur_xlim = self.ax.get_xlim()
            self.cur_ylim = self.ax.get_ylim()
            self.xpress, self.ypress = event.xdata, event.ydata
            self.moved = False

        def onRelease(event):
            self.xdrag, self.ydrag = False, False
            if self.axt is not None and self.drag_axes==self.ax and not self.moved:
                self.update_info(event.xdata, event.ydata)
            self.ax.figure.canvas.draw()

        def onMotion(event):
            if not (self.xdrag or self.ydrag): return
            self.moved = True
            if event.inaxes != self.drag_axes: return
            if self.xdrag:
                dx = event.xdata - self.xpress
                self.cur_xlim -= dx
                self.ax.set_xlim(self.cur_xlim)
            if self.ydrag:
                dy = event.ydata - self.ypress
                self.cur_ylim -= dy
                self.ax.set_ylim(self.cur_ylim)
            self.sync_panels_scale()
            self.ax.figure.canvas.draw()

        fig = self.ax.get_figure()
        fig.canvas.mpl_connect('button_press_event',  onPress)
        fig.canvas.mpl_connect('button_release_event',onRelease)
        fig.canvas.mpl_connect('motion_notify_event', onMotion)

    def update_info(self, x, y, fun=None):
        if self.axt is None: return
        if fun is None:
            if self.text_fun is None:
                print("Warning: unavailable text-box handler.")
                return
            fun = self.text_fun
        self.axt.clear()
        self.axt.set_xticks(())
        self.axt.set_yticks(())
        fun(self.axt, x, y)


def demo_display_info(ax, x, y):
    return ax.text(0.05, 0.95, f"Hello({x:.1f},{y:.1f})!", transform=ax.transAxes,
                   family='monospace', fontsize=10, verticalalignment='top')


if __name__ == "__main__":
    # configuration
    N = 1000

    # full version
    z = InteractiveFigure(text_panel=True, text_fun=demo_display_info)
    ax1 = z.get_axes()
    ax1.plot(list(range(N)), np.random.rand(N), 'r-')
    ax1.set_title("Title")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # skinny version
    z = InteractiveFigure()
    ax2 = z.get_axes()
    ax2.plot(list(range(N)), np.random.rand(N), 'r-')
    ax2.set_title("Skinny")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    # pause
    plt.show()
