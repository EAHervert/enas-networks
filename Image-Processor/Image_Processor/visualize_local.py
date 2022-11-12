import torch
import os


class Visualizer:
    def __int__(self,
                path='/Users/esauhervert/PycharmProjects/enas-networks/Image-Processor/images/SIDD_Medium_Srgb'):

        self.path = path

    def print_list(self):
        for i in os.listdir(self.path):
            print(i)
