import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import argparse

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from generator import Generator
from instances import PixelInstance
from models import CNN1, CNN2, CNN3, NNPixelDataset, SkipCNN1, UpAE
from utils import get_device, objdict, label2heatmap
from tester import Tester


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


def test_example(tester, generator, size, population, number, output_type, verbose=False, show_fail=False):
    instances = [generator.get_pixel_instance() for i in range(number)]

    all_images = []
    sharp_accuracy = []
    nearest_accuracy = []
    pointing_accuracy = []

    with torch.no_grad():
        for i in range(number):
            # Prepare data
            image = np.asarray(instances[i].image).astype(np.uint8)
            mini_batch = torch.tensor(image / 255).unsqueeze(0).unsqueeze(0).float().to(device)
            label = torch.tensor(instances[i].neighbor_list[0]).to(device)

            if output_type=='map':
                labels = label2heatmap(label.unsqueeze(0), 50).to(device)
                labels = torch.argmax(labels, 1)
            else :
                labels = label.float()
            #
            # Run model
            output = tester.model(mini_batch)

            loss = tester.criterion(output, labels)

            if output_type=='coord':
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
                pred_x, pred_y = np.clip(rounded_output[0].cpu().int().numpy(), 0, size-1)
                result_image = np.stack((instances[i].image, instances[i].image, instances[i].image), axis=2)
                if instances[i].image[pred_x, pred_y] :
                    result_image[pred_x, pred_y] = [0,1,0]
                else :
                    result_image[pred_x, pred_y] = [1,0,0]

            elif output_type=='map':
                predictions = torch.argmax(output, 1)
                correct = (predictions == labels).float().sum()
                sharp_accuracy.append(correct)
                nearest_accuracy.append(0)
                pointing_accuracy.append(0)
                prediction_image = output.reshape([size, size])
                result_image = np.stack((instances[i].image, instances[i].image, prediction_image.cpu()), axis=2)

            all_images.append(result_image)

            if verbose or (show_fail and correct==0):
                print(' - output:', output, ' - ')
                # print(' - Rounded: ', rounded_output, ' - ')
                print(' - neighbor list:', instances[i].neighbor_list)
                print(' - loss:', loss, ' - ')
                print(' - Corectness: ', correct, ' - ')
                n=2
                pred_im = prediction_image.cpu().numpy()
                print('- Prediction argmax: ', np.unravel_index(pred_im.argmax(), pred_im.shape))
                normalized = (pred_im - pred_im.min() ) / (pred_im.max() - pred_im.min())
                print('minmax : ', normalized.max(), normalized.min())
                m = torch.nn.Softmax()
                soft_prediction = m(prediction_image).cpu()
                f = plt.figure()
                # f.add_subplot(1,n, 1)
                # plt.imshow(np.clip(pred_im, 0, 1))
                f.add_subplot(1,n, 1)
                plt.imshow(normalized)
                # f.add_subplot(1,n, 3)
                # plt.imshow(soft_prediction)
                f.add_subplot(1,n, 2)
                plt.imshow(instances[i].image)
                plt.show()

    if number > 1:
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

    # config = objdict({ #Test11-13-30
    #       "alias": "Test",
    #       "batch_size": 128,
    #       "checkpoint_dir": "",
    #       "checkpoint_type": "best",
    #       "criterion": "crossentropy",
    #       "data": "/home/ardoin/Dial-a-Ride/data/instances/split3_1nn_500k_n2_s50_m0",
    #       "epochs": 2000,
    #       "gamma": 0.1,
    #       "input_type": "map",
    #       "lr": 0.001,
    #       "milestones": [
    #         50
    #       ],
    #       "model": "UpAE",
    #       "optimizer": "Adam",
    #       "output_type": "map",
    #       "patience": 50,
    #       "scheduler": "plateau",
    #       "seed": 836569618,
    #       "shuffle": True,
    #       "layers": 128
    # })
    config = objdict({
          "alias": "AE",
          "batch_size": 128,
          "channels": 2,
          "checkpoint_dir": "",
          "checkpoint_type": "best",
          "criterion": "crossentropy",
          "data": "/scratch/izar/ardoin/split3_1nn_300k_n2_s40_m1",
          "epochs": 200,
          "file_dir": "/home/ardoin/experiments_obs",
          "gamma": 0.1,
          "input_type": "map",
          "layers": 256,
          "lr": 0.0001,
          "milestones": [
            50
          ],
          "model": "UpAE",
          "optimizer": "Adam",
          "output_type": "map",
          "patience": 10,
          "scheduler": "plateau",
          "seed": 743871277,
          "shuffle": True,
          "upscale_factor": 2
    })
    device = get_device()

    tester = Tester(config, saved_model='./data/experiments/distant/experiments_obs/23/best_model.pt')

    generator = Generator(size=40, population=2, moving_car=True)

    test_example(tester, generator, size=50, population=2, number=1000, output_type=config.output_type, verbose=True, show_fail=False)
