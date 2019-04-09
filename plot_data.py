import sys
import json
import logging
import requests
import threading
import time
import websocket # 0.48.0
import pkg_resources

from signalk_client.client import Client, Data
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# signal K client
sk_client = Client(server='192.168.0.108:3000')
# init vessel comm 
vessel = sk_client.data.get_self()

# define path
path = 'navigation.headingMagnetic'
# path = 'steering.rudderAngle'
# path = 'navigation.speedOverGround'
# path = 'environment.wind.angleApparent'
#path = 'environment.wind.speedApparent'

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
time = []
data = []
datadot = []

# This function is called periodically from FuncAnimation
def animate(i, time, data, datadot):

    # Add x and y to lists
    data.append(vessel.get_prop(path)['value'])
    time.append(dt.datetime.strptime(vessel.get_prop(path)['timestamp'], '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())
    if len(data) >5:
        datadot.append((data[-1] - data[-4]) / (time[-1] - time[-4]))
    else:
        datadot.append(0)
        
    # Limit x and y lists to 20 items
    time = time[-20:]
    data = data[-20:]
    datadot = datadot[-20:]
    
    # Draw x and y lists
    ax.clear()
    ax.plot(time, datadot)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('data over Time')
    plt.ylabel('data')

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(time, data, datadot), interval=500)
plt.show()
