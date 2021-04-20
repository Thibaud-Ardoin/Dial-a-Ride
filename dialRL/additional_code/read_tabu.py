import csv
import torch
import pandas as pd
import drawSvg as draw
import matplotlib.pyplot as plt

file_name = './data/tabu6.csv'

instance = pd.read_csv(file_name, sep='\s+', squeeze=True, header=None)

instance.columns = ['id', 'x', 'y', '10', 'type', 'number', 'other_number']

print(instance)

print('Min and max ! -> ', max(instance['x']) - min(instance['x']), max(instance['y']) - min(instance['y']))
MaxX, MaxY = max(abs(instance['x'])), max(abs(instance['y']))
size=max(MaxX, MaxY)

d = draw.Drawing(size*2, size*2, origin='center', displayInline=False)

for elmt in range(len(instance)):
    d.append(draw.Circle(instance['x'][elmt], instance['y'][elmt], 0.2,
            fill='red', stroke_width=0.1, stroke='black'))

for elmt in range(len(instance) // 2):
    d.append(draw.Line(instance['x'][elmt], instance['y'][elmt],
                       instance['x'][len(instance)-elmt-1], instance['y'][len(instance)-elmt-1],
                       stroke='green', stroke_width=0.02, fill='none'))

# d.append(draw.Circle(size,0, 0.5,
#         fill='blue', stroke_width=0.1, stroke='black'))
#
# d.append(draw.Circle(0, 0, 0.5,
#         fill='blue', stroke_width=0.1, stroke='black'))
#
# d.append(draw.Circle(0, size, 0.5,
#         fill='blue', stroke_width=0.1, stroke='black'))

#d.setPixelScale(2)  # Set number of pixels per geometry unit
d.setRenderSize(size*100, size*100)
d.saveSvg('test.svg')
d.savePng('test.png')
