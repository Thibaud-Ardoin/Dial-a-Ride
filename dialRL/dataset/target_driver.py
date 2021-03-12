
class Target():
    def __init__(self, pickup, dropoff, start, end, identity, weight=1):
        self.pickup = pickup
        self.dropoff = dropoff
        self.start_fork = start
        self.end_fork = end
        self.weight = weight
        self.identity = identity

        # State is in [-1, 0, 1] for [wait pick up, in a car, done]
        self.state = -1
        self.available = 0

    def __repr__(self):
        return "Target: " + str(self.pickup) + ' to ' + str(self.dropoff)

    def __str__(self):
        return "Target: " + str(self.pickup) + ' to ' + str(self.dropoff)

    def start_in_time(self, current_time):
        if self.start_fork[0] <= current_time :
            if self.start_fork[1] >= current_time:
                return True
        return False

    def end_in_time(self, current_time):
        if self.end_fork[0] <= current_time :
            if self.end_fork[1] >= current_time:
                return True
        return False

    def get_info_vector(self):
        vector = [self.identity]
        vector.append(self.pickup[0])
        vector.append(self.pickup[1])
        vector.append(self.dropoff[0])
        vector.append(self.dropoff[1])
        vector.append(self.start_fork[0])
        vector.append(self.start_fork[1])
        vector.append(self.end_fork[0])
        vector.append(self.end_fork[1])
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

    def load(self, target, current_time):
        if target.weight + self.capacity() > self.max_capacity :
            return False
        else :
            if not target.start_in_time(current_time):
                return False
            else :
                self.loaded.append(target)
                return True

    def unload(self, target, current_time):
        indice = target.identity
        for i,t in enumerate(self.loaded):
            if t.identity == indice:
                if t.end_in_time(current_time):
                    del self.loaded[i]
                    return True
                else :
                    return False
        return False

    def is_in(self, indice):
        for target in self.loaded:
            if target.identity == indice:
                return True
        return False

    def get_info_vector(self):
        vector = [self.identity, self.position[0], self.position[1]]
        loaded_target_id = [self.loaded[i].identity for i in range(len(self.loaded))]
        countainer = loaded_target_id + [0.] * (self.max_capacity - len(self.loaded))
        return vector + countainer
