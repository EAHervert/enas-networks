import utilities.dataset_helper_functions as functions
import utilities.dataloader_images as dataloader_

test = functions.image_csv()
dataloader_obj = dataloader_.load_dataset_images(test.training_instances)

for index, (x, y) in enumerate(dataloader_obj):
    print(x.size(), y.size())
