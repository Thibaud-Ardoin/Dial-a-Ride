import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import argparse

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from generator import Generator, PixelInstance
from models import CNN1, CNN2, CNN3, NNPixelDataset, SkipCNN1
from utils import get_device


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Parse argument used when running a train.",
        epilog="python train.py --epochs INT")

    # required input parameters
    parser.add_argument(
        '--name', type=str,
        help='Name from the model experiment. Ex: testing09-15-14')

    parser.add_argument(
        '--model', type=str,
        help='Model name, like CNN1')

    return parser.parse_known_args(args)[0]


def test_example(model, generator, size, population, number, verbose=False):
    instances = [generator.get_pixel_instance(size=size, population=population) for i in range(number)]

    criterion = torch.nn.MSELoss()
    all_images = []
    sharp_accuracy = []
    nearest_accuracy = []
    pointing_accuracy = []

    with torch.no_grad():
        for i in range(number):
            # Prepare data
            image = np.asarray(instances[i].image).astype(np.uint8)
            mini_batch = torch.tensor(image / 255).unsqueeze(0).unsqueeze(0).float().to(device)
            label = torch.tensor(instances[i].neighbor_list[0]).float().to(device)

            # Run model
            output = model(mini_batch)
            loss = criterion(output, label)

            # Compute results
            rounded_output = torch.round(output)

            # Perfect guessed match
            correct = (rounded_output == label).all().sum().item()
            sharp_accuracy.append(correct)

            distance_pred2points = list(map(lambda x: np.linalg.norm(rounded_output.cpu() - x), instances[i].neighbor_list))
            # Case where the model aims perfectly to one pixel
            pointing_accuracy.append(np.min(distance_pred2points) == 0)

            # Case where the nearest pixel to prediction is the nearest_neighbors
            nearest_accuracy.append(np.argmin(distance_pred2points) == 0)

            if verbose :
                print(' - output:', output, ' - ')
                print(' - Rounded: ', rounded_output, ' - ')
                print(' - neighbor list:', instances[i].neighbor_list)
                print(' - MSE loss:', loss, ' - ')
                print(' - Corectness: ', correct, ' - ')

            pred_x, pred_y = np.clip(rounded_output[0].cpu().int().numpy(), 0, size-1)
            result_image = np.stack((instances[i].image, instances[i].image, instances[i].image), axis=2)
            if instances[i].image[pred_x, pred_y] :
                result_image[pred_x, pred_y] = [0,1,0]
            else :
                result_image[pred_x, pred_y] = [1,0,0]
            all_images.append(result_image)

    fig=plt.figure(figsize=(20, 20))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    columns = 3
    rows = 2
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(all_images[i-1])
        ax = fig.gca()
        ax.set_axis_off()
    plt.show()

    # Compile accuracies:
    print(' - Sharp Accuracy: ', 100*np.sum(sharp_accuracy)/len(sharp_accuracy), '%')
    print(' - Pointing Accuracy: ', 100*np.sum(pointing_accuracy)/len(pointing_accuracy), '%')
    print(' - Nearest Accuracy: ', 100*np.sum(nearest_accuracy)/len(nearest_accuracy), '%')



if __name__ == '__main__':

    flags = parse_args(sys.argv[1:])

    device = get_device()

    try :
        model = locals()[flags.model]().to(device)
    except:
        raise "The model as input has not been found !"
    # if flags.model == 'CNN1' :
    #     model = CNN1().to(device)
    # if flags.model == 'CNN2' :
    #     model = CNN2().to(device)
    # if flags.model == 'CNN3' :
    #     model = CNN3().to(device)

    model.load_state_dict(torch.load('./data/experiments/' + flags.name + '/best_model.pt'))
    model.eval()

    generator = Generator()

    test_example(model, generator, size=50, population=2, number=10000)
