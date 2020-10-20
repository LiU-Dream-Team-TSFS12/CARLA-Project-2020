import matplotlib.pyplot as plt
import time

def BoxOff(*argin):
    if len(argin)>0:
        ax=argin[0]
    else:
        ax=plt.gca();
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


class Timer:
    t0 = 0.0
    dt = 0.0
    def tic(self):
        self.t0 = time.time()

    def toc(self):
        self.dt = time.time()-self.t0
        return self.dt
