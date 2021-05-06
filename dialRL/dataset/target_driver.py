from dialRL.utils import distance, float_equality, coord2int


class Target():
    def __init__(self, pickup, dropoff, start, end, identity, weight=1):
        self.pickup = pickup
        self.dropoff = dropoff
        self.start_fork = start
        self.end_fork = end
        self.weight = weight
        self.identity = identity

        # State is in [-2, -1, 0, 1, 2] for
        # [wait pick up, getting picked up, In car, getting dropped, done]
        self.state = -2
        self.available = 0

    def __repr__(self):
        return "Target: " + str(self.pickup) + ' to ' + str(self.dropoff)

    def __str__(self):
        return "Target: " + str(self.pickup) + ' to ' + str(self.dropoff)

    def start_in_time(self, current_time):
        if float_equality(current_time, self.start_fork[0]) or float_equality(current_time, self.start_fork[1]) :
            return True
        if self.start_fork[0] <= current_time :
            if self.start_fork[1] >= current_time:
                return True
        return False

    def end_in_time(self, current_time):
        if float_equality(current_time, self.end_fork[0]) or float_equality(current_time, self.end_fork[1]) :
            return True
        if self.end_fork[0] <= current_time :
            if self.end_fork[1] >= current_time:
                return True
        return False

    def get_info_vector(self):
        vector = [self.identity]
        vector.append(coord2int(self.pickup[0]))
        vector.append(coord2int(self.pickup[1]))
        vector.append(coord2int(self.dropoff[0]))
        vector.append(coord2int(self.dropoff[1]))
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
        self.destination = None
        self.target = None
        self.order = 'waiting'
        self.history_move = [position]
        self.loaded = [] #Target list

    def __repr__(self):
        return "Driver at : " + str(self.position) + ' he is ' + self.order

    def __str__(self):
        return "Driver at : " + str(self.position) + ' he is ' + self.order

    def next_available(self, current_time):
        if self.destination is None:
            return current_time
        else :
            return current_time + distance(self.destination, self.position)


    def can_load(self, target, current_time):
        if target.weight + self.capacity() > self.max_capacity :
            return False
        else :
            if target.start_in_time(current_time + distance(target.pickup, self.position)):
                return True
            else :
                return False


    def can_unload(self, target, current_time):
        indice = target.identity
        for i,t in enumerate(self.loaded):
            if t.identity == indice:
                if t.end_in_time(current_time + distance(target.dropoff, self.position)):
                    return True
                else :
                    return False
        return False


    def set_target(self, target, current_time):
        if target is None:
            self.target = None
            self.destination = None
            self.order = 'waiting'

        else :
            if self.target is not None:
                print('t id:', target.identity, self.target.identity)
                raise ValueError("Setting up a new target without having delivered the privious one ! ")
            # Set dropping off target
            if target.state == 0 :
                if self.can_unload(target, current_time) :
                    self.destination = target.dropoff
                    self.target = target
                    self.order = 'dropping'
                    target.state = 1
                    return True
                else :
                    return False

            # Set picking up target
            elif target.state == -2 :
                if self.can_load(target, current_time):
                    self.target = target
                    self.order = 'picking'
                    target.state = -1
                    self.destination = target.pickup
                    return True
                else :
                    return False
            else :
                raise "Error with setting outdated target: " + str(target)


    def move(self, new_position):
        self.history_move.append(new_position)
        self.distance = distance(new_position, self.position)
        # print('Driver ', self.identity, 'Moves from : ', self.distance)
        self.position = new_position

    def capacity(self):
        c = 0
        for target in self.loaded:
            c += target.weight
        return c

    def load(self, target, current_time):
        if target.weight + self.capacity() > self.max_capacity :
            print('Weight problem')
            return False
        else :
            if not target.start_in_time(current_time):
                print(current_time, target.start_fork, target.end_fork)
                print('Time windows problem')
                return False
            else :
                target.state = 0
                self.loaded.append(target)
                return True

    def unload(self, target, current_time):
        indice = target.identity
        for i,t in enumerate(self.loaded):
            if t.identity == indice:
                if t.end_in_time(current_time):
                    target.state = 2
                    del self.loaded[i]
                    return True
                else :
                    print(current_time, target.start_fork, target.end_fork)
                    print('Time windows problem')
                    return False
        return False

    def is_in(self, indice):
        for target in self.loaded:
            if target.identity == indice:
                return True
        return False

    def get_info_vector(self):
        vector = [self.identity, coord2int(self.position[0]), coord2int(self.position[1])]
        loaded_target_id = [self.loaded[i].identity for i in range(len(self.loaded))]
        countainer = loaded_target_id + [0.] * (self.max_capacity - len(self.loaded))
        return vector + countainer
