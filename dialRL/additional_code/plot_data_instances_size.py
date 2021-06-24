import matplotlib.pyplot as plt
import numpy as np

file_name = './dialRL/additional_code/dataset_size'
X1 = []
X2 = []

with open(file_name) as f:
    line = ''
    while line != 'EOF' :
        line = f.readline()[:-2]
        if line == '':
            break

        data = line.split(', ')
        if int(data[2]) == 1 :
            X1.append([int(data[0]), int(data[1])])
        else :
            X2.append([int(data[0]), int(data[1])])

X1 = np.array(X1)
X2 = np.array(X2)

plt.scatter(X1[:, 0], X1[:, 1], color='blue', label='Cordeau 2006')
plt.scatter(X2[:, 0], X2[:, 1], color='red', label='Cordeau 2003')
plt.legend()
plt.Xtitle('Number of drivers')
plt.Ytitle('Number of targets')

plt.show()
