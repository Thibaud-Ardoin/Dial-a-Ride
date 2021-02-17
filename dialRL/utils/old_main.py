from generator import Generator, PixelInstance

import numpy as np
import time
import sys

if __name__ == '__main__':
    generator = Generator()
    start_t = time.time()
    instance = generator.get_pixel_instance(size=10, population=5)
    end_t = time.time()

    print('size of object: ', sys.getsizeof(instance))
    print('size of image alone: ', sys.getsizeof(instance.image))

    print('Needed time in sec: ', end_t - start_t)
    print('Needed time for 1.000.000 images: ', 1000000*(end_t - start_t))
    instance.reveal()
