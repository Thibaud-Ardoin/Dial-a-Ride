import numpy as np
from utils import distance, float_equality

a, b = np.random.uniform(9, -9), np.random.uniform(9, -9)
a, b = np.max(a, b), np.min(a, b)

print(a, b)
lam = b / a

origin = np.array([1. , 1.])
target = np.array([2., 2.])

new_pos = (1 - lam) * origin + (lam) * target

print('distance from on to other')

if not float_equality(distance(new_pos, driver.position), self.last_time_gap, eps=0.01):
