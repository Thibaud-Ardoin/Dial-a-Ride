
class Target():
    def __init__(self, pickup, dropoff, start, end, identity, weight=0):
        self.pickup = pickup
        self.dropoff = dropoff
        self.start = start
        self.end = end
        self.weight = weight
        self.identity = identity

        # State is in [-1, 0, 1] for [wait pick up, in a car, done]
        self.state = -1
        self.available = 0

    def __repr__(self):
        return "Target: " + str(self.pickup) + ' to ' + str(self.dropoff)

    def __str__(self):
        return "Target: " + str(self.pickup) + ' to ' + str(self.dropoff)

    def get_info_vector(self):
        vector = [self.identity]
        vector.append(self.pickup[0])
        vector.append(self.pickup[1])
        vector.append(self.dropoff[0])
        vector.append(self.dropoff[1])
        vector.append(self.start[0])
        vector.append(self.start[1])
        vector.append(self.end[0])
        vector.append(self.end[1])
        vector.append(self.weight)
        vector.append(self.state)
        return vector


class Driver():
    def __init__(self, position, identity, max_capacity=6, speed=1, verbose=False):
        self.position = position
        self.max_capacity = max_capacity
        self.speed = speed
        self.identity = identity
        self.distance = 0
        self.loaded = [] #Target list

    def __repr__(self):
        return "Driver at : " + str(self.position)

    def __str__(self):
        return "Driver at : " + str(self.position)

    def move(self, new_position):
        self.distance = distance(new_position, self.position)
        self.position = new_position

    def capacity(self):
        c = 0
        for target in self.loaded:
            c += target.weight
        return c

    def load(self, target):
        if target.weight + self.capacity() > self.max_capacity :
            return False
        else :
            self.loaded.append(target)
            return True

    def unload(self, target):
        indice = target.identity
        for i,t in enumerate(self.loaded):
            if t.identity == indice:
                del self.loaded[i]
                return True
        return False

    def is_in(self, indice):
        for target in self.loaded:
            if target.identity == indice:
                return True
        return False

    def get_info_vector(self):
        vector = [self.identity, self.position[0], self.position[1]]
        countainer = self.loaded + [0.] * (self.max_capacity - len(self.loaded))
        return vector + countainer
