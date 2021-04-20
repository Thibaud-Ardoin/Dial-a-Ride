import csv
# import torch
import pandas as pd
import drawSvg as draw
import matplotlib.pyplot as plt
import re

file_name = './data/tabu1.csv'
instance = pd.read_csv(file_name, sep='\s+', squeeze=True, header=None)
instance.columns = ['id', 'x', 'y', '10', 'type', 'number', 'other_number']

print('Min and max ! -> ', max(instance['x']) - min(instance['x']), max(instance['y']) - min(instance['y']))
MaxX, MaxY = max(abs(instance['x'])), max(abs(instance['y']))
size=max(MaxX, MaxY)

d = draw.Drawing(size*2, size*2, origin='center', displayInline=False)

print(instance.head())

file_name = './res1.txt'
with open(file_name) as data:
    text = data.read()

blocks = text.split('\n\n')[1:-1]


# Draw data instance
d = draw.Drawing(size*2, size*2, origin='center', displayInline=False)

for elmt in range(len(instance)):
    d.append(draw.Circle(instance['x'][elmt], instance['y'][elmt], 0.2,
            fill='red', stroke_width=0.1, stroke='black'))

for elmt in range(len(instance) // 2):
    d.append(draw.Line(instance['x'][elmt], instance['y'][elmt],
                       instance['x'][len(instance)-elmt-1], instance['y'][len(instance)-elmt-1],
                       stroke='black', stroke_width=0.02, fill='none'))

d.append(draw.Circle(float(instance[instance['id'] == 0]['x']), float(instance[instance['id'] == 0]['y']), 0.3,
                     fill='deeppink', stroke_width=0.2, stroke='black'))

print('ZEROOOO', instance[instance['id'] == 0], 'aha', float(instance['x'][instance['id'] == 0]), float(instance['y'][instance['id'] == 0]))

# Build and draw the drivers
colors = ['blueviolet', 'seagreen', 'orangered', 'deepskyblue']
drivers = []
for k, block in enumerate(blocks):
    driver_data = block.split('\n')
    driver_info = re.split('\s+', driver_data[0])
    driver = {}
    driver['id'] = driver_info[0]
    for i in [1,3,5,7] :
        driver[driver_info[i][0]] = float(driver_info[i+1])
    driver['order'] = [int(driver_data[i].split(' ')[0]) for i in range(1, len(driver_data))]
    drivers.append(driver)

    driver['path'] = []
    for node in driver['order']:
        this_node = instance[instance['id'] == node]
        driver['path'].append([float(this_node['x']), float(this_node['y'])])

    print(driver)
    for edge in range(1,len(driver['path'])):
        d.append(draw.Line(driver['path'][edge-1][0], driver['path'][edge-1][1],
                           driver['path'][edge][0], driver['path'][edge][1],
                           stroke=colors[k], stroke_width=0.1, fill='none'))

d.setRenderSize(size*100, size*100)
d.saveSvg('results.svg')
d.savePng('results.png')
