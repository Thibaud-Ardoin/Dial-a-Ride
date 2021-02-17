import pickle
from generator import PixelInstance

if __name__ == '__main__':

    filehandler = open('./data/instances/split3_1nn_25k_n2_s50/test_instances.pkl', 'rb')
    instances = pickle.load(filehandler)
    filehandler.close()

    cnt = 0
    for instance in instances :
        if len(instance.nearest_neighbors) != 1 :
            cnt += 1
            #instance.reveal())

    print('Percentage of equal nearest neighbor : ', 100 * cnt / len(instances))
    print(' Count of double nn : ', cnt)
