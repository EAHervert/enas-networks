import json
import torch
import utilities.utils
import utilities.functions as functions
import utilities.dataloader_images as dataloader_


config_path = '/Users/esauhervert/PycharmProjects/enas-networks/config.json'

test = functions.image_csv(config_path=config_path)
functions.create_batches(test.)