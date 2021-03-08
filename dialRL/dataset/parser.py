from dialRL.dataset import Target, Driver


def tabu_parse(file_name):
    targets = []
    drivers = []
    with open(file_name, 'r') as file :
        nb_drivers, number_line, c, e, f = list(map(int, file.readline().split()))
        # print("What is that ? :", c, e, f)

        # Depot
        identity, X, Y, we, ty, st, en = list(map(float, file.readline().split()))
        for d in range(nb_drivers):
            driver = Driver(position=(X, Y), identity=d+1, max_capacity=e, speed=1, verbose=False)
            drivers.append(driver)

        for l in range(number_line) :
            if l < number_line//2 :
                identity, X, Y, we, ty, st, en = list(map(float, file.readline().split()))
                # print(identity, X, Y, we, ty, st, en)
                t = Target(pickup=(X, Y), dropoff=None, start=(st, en), end=None, identity=int(identity), weight=1)
                targets.append(t)
            else :
                identity, X, Y, we, ty, st, en = list(map(float, file.readline().split()))
                re_t = targets[l - number_line//2]
                re_t.dropoff = (X, Y)
                re_t.end = (st, en)

    return targets, drivers

def check_extrema(extremas, X, Y):
    mx = min(X, extremas[0])
    my = min(Y, extremas[1])
    Mx = max(X, extremas[2])
    My = max(Y, extremas[3])
    return mx, my, Mx, My

def tabu_parse_info(file_name):
    extremas = [0, 0, 0, 0]
    with open(file_name, 'r') as file :
        nb_drivers, number_line, c, e, f = list(map(int, file.readline().split()))
        identity, X, Y, we, ty, st, en = list(map(float, file.readline().split()))

        # Compute Infos
        time_end = en
        target_population = number_line // 2
        driver_population = nb_drivers
        depot_position = (X, Y)

        extremas = check_extrema(extremas, X, Y)
        for l in range(number_line) :
            identity, X, Y, we, ty, st, en = list(map(float, file.readline().split()))
            extremas = check_extrema(extremas, X, Y)
    size = max(abs(extremas[2] - extremas[0]), abs(extremas[3] - extremas[1]))
    return extremas, target_population, driver_population, time_end, depot_position, size


if __name__ == '__main__':
    targets, drivers = parse('./data/instances/cordeau2003/tabu1.txt')
    for t in targets :
        print(t)
    for d in drivers :
        print(d)