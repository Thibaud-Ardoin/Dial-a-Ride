import numpy as np
from dialRL.utils import distance, float_equality

origin = np.array([1. , 1.])
target = np.array([10., 10.])

print('origin position: ', origin)
print('Traget position: ', target)

a = distance(origin, target)
b = np.random.uniform(0, 9)

# , b =  np.random.uniform(0, 9)
# a, b = np.max([a, b]), np.min([a, b])
lam = b / a

print('Distance to target(a): ', a)
print('Current time gap(b): ', b)

print('Current proportion of advance on the tragetory (lam):', lam)


new_pos = (1 - lam) * origin + (lam) * target

print('New position : ', new_pos)

to_go_now = distance(new_pos, origin)
to_go_later = distance(new_pos, target)

print('Distance from origin to new_position: ', to_go_now)
print('Distance from new_position to target: ', to_go_later)
print('Addition of distances: ', to_go_later + to_go_now)
eps = 0.0001

compare_gap_distance = float_equality(distance(new_pos, origin), b, eps=eps)

print('Comparaison float  equality: ', compare_gap_distance, 'with eps=', eps)
