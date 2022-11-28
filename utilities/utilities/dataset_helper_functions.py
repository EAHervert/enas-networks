import os
import json
import pandas as pd  # Saving CSV
import random


class image_csv:

    def __init__(self,
                 config_path='/Users/esauhervert/PycharmProjects/enas-networks/config.json'):

        # Console parameter specification
        self.config_path = config_path
        self.config = json.load(open(self.config_path))

        self.path = self.config['Locations']['Dataset']

        self.files = os.listdir(self.path)
        self.pairs = []

        for file in self.files:
            self.pairs.append(self.extract_pairs(os.listdir(self.path + '/' + file), self.path + '/' + file))

        self.percentage = self.config['Training']['Train_Percentage']

        self.training_size = 0
        self.validation_size = 0
        self.data = []
        index = 0
        for entry in self.pairs:
            for ent_i in [entry['010'], entry['011']]:
                temp = {'INDEX': index, 'INPUT': ent_i['NOISY'], 'TARGET': ent_i['GT']}
                if self.select(self.percentage):
                    temp['SET'] = "Training"
                else:
                    temp['SET'] = "Validation"

                index += 1
                self.data.append(temp)

        self.data_csv = pd.DataFrame(self.data, columns=['INDEX', 'INPUT', 'TARGET', 'SET'], index=None)

        self.training_csv = self.data_csv[self.data_csv['SET'] == 'Training']
        self.validation_csv = self.data_csv[self.data_csv['SET'] == 'Validation']

        self.training_instances = [{'INPUT': item['INPUT'],
                                    'TARGET': item['TARGET']} for _, item in self.training_csv.iterrows()]

        self.validation_instances = [{'INPUT': item['INPUT'],
                                      'TARGET': item['TARGET']} for _, item in self.validation_csv.iterrows()]

    @staticmethod
    def extract_pairs(paths, parent):
        dict_paths = {"010": {},
                      "011": {}}

        for path in paths:
            if '010.PNG' in path:
                if 'NOISY' in path:
                    dict_paths["010"]['NOISY'] = parent + '/' + path
                else:
                    dict_paths["010"]['GT'] = parent + '/' + path
            else:
                if 'NOISY' in path:
                    dict_paths["011"]['NOISY'] = parent + '/' + path
                else:
                    dict_paths["011"]['GT'] = parent + '/' + path

        return dict_paths

    @staticmethod
    def select(percent=0.5):
        return random.randrange(100) < percent * 100
