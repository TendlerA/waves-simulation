#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 10:52:02 2018
SImulation of waves
@author: avichai
"""



"""
Matplotlib Animation Example

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
line, = ax.plot([], [],'.')

#initialize parameters
num_frames=200
num_points=3000
points=np.random.rand(num_points,2)
d_dist=np.sqrt(np.subtract.outer(points[:,0],points[:,0])**2+np.subtract.outer(points[:,1],points[:,1])**2)
values=np.zeros([num_points,num_frames+1])

for i in range(num_frames):
    #start from prev. activation values
    values[:,i+1]=values[:,i]
    
    #add random excitations
    act_prob=1/10
    if np.random.rand()>1-act_prob:
        values[np.random.randint(0,num_points),i+1]=1

    #Local activation, activate if there is a neighbor
    thres_dist=1/20
    repvals=np.tile(values[:,i],(num_points,1))
    m=repvals/(d_dist+0.001)
    values[np.any(m>1/thres_dist,axis=1),i+1]=1
    
    #Global inhibition, delete all if there are too many activated points
    thres_inhibition=300
    if sum(values[:,i+1])>thres_inhibition:
        values[:,i+1]=0

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i,values=values,points=points):
    #plot only active points
    line.set_data(points[values[:,i+1]==1,0], points[values[:,i+1]==1,1])
    
    return line,

def plot_total_activation(values):
    total_act=np.sum(values,axis=0);
    plt.figure()
    plt.plot(total_act)
    plt.xlabel('Time')
    plt.ylabel('Active Cells #')
    
# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=num_frames, interval=100, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('basic_animation.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

plt.show()

plot_total_activation(values)