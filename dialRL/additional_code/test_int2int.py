import math
from icecream import ic


extremas = [-4, -4, 4, 4]


src_vocab_size = 70000
siderange = int(math.sqrt(src_vocab_size))
ic(siderange)
boxh = abs(extremas[2] - extremas[0]) / siderange
boxw = abs(extremas[3] - extremas[1]) / siderange
ic(boxh, boxw)


def bidim2int(coordonate):
    h = int(abs(coordonate[0] - extremas[0]) / boxh )
    w = int(abs(coordonate[1] - extremas[1]) / boxw )
    return int(h + (w * siderange))

def int2bidim(int_position):

    h = int_position - siderange * (int_position // siderange)
    w = (int_position - h) / siderange
    h = (h * boxh) + extremas[0]
    w = (w * boxw) + extremas[1]
    return h, w

coord = [-3.356, -1.674]

ic(bidim2int(coord))
ic(int2bidim(bidim2int(coord)))
