# Dataloader Images
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.multiprocessing

# To allow opening the files I need to open.
torch.multiprocessing.set_sharing_strategy('file_system')


# This class is a dataset constructed from one image.
class SIDD_Medium_Dataset_Images(Dataset):
    def __init__(self, image_instances, size_crop, size_n, size_m, transform=None):
        self.image_instances = image_instances
        self.size_crop = size_crop
        self.size_n = size_n
        self.size_m = size_m
        self.transform = transform

        # List of tensors:
        self.tensor_ni = torch.zeros(
            len(self.image_instances),
            self.size_n,
            self.size_m,
            3,
            self.size_crop,
            self.size_crop
        )
        self.tensor_gt = torch.zeros(
            len(self.image_instances),
            self.size_n,
            self.size_m,
            3,
            self.size_crop,
            self.size_crop
        )
        self.j = 0
        # Load the images to a list of tensors:
        for i in self.image_instances:
            # We will first load the whole batch tensors:
            location_noise = i[0]
            tensor_ni = torch.load(location_noise)

            location_gt = i[1]
            tensor_gt = torch.load(location_gt)

            # Now, we define the size of the tensor images.
            n = int(i[2][0] + i[2][1])
            m = int(i[3][0] + i[3][1])

            # Permutations needed:
            perm_n = torch.randperm(n)[0:self.size_n]
            perm_m = torch.randperm(m)[0:self.size_m]

            tensor_ni = tensor_ni.index_select(0, perm_n).index_select(1, perm_m)
            tensor_gt = tensor_gt.index_select(0, perm_n).index_select(1, perm_m)

            self.tensor_ni[self.j, :, :, :, :, :] = tensor_ni
            self.tensor_gt[self.j, :, :, :, :, :] = tensor_gt

            self.j += 1

    def __len__(self):
        return self.j * self.size_n * self.size_m

    def __getitem__(self, index):
        i = index // (self.size_n * self.size_m)
        index = index % (self.size_n * self.size_m)
        n = index // self.size_m
        m = index // self.size_n

        tensor_in = self.tensor_ni[i, n, m, :, :, :]
        tensor_out = self.tensor_gt[i, n, m, :, :, :]

        return tensor_in, tensor_out


# This function will go through all the images and select random batches for training and validation.
# The number of random batches is given by Num_Image_batches
def load_dataset_images(
        instances,
        batch_size=16,
        size_crop=64,
        size_n=32,
        size_m=32
):
    dataset = SIDD_Medium_Dataset_Images(instances, size_crop, size_n, size_m)

    # Create Dataloader:
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_loader
