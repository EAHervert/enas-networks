from Image_Processor import image_crop
import os


def image_crops(path, size=64, sample_x=5, sample_y=5):
    vis = image_crop.Crops(path=path)

    vis.set_random_images()
    vis.get_random_crops(size=size, sample_x=sample_x, sample_y=sample_y)
    vis.set_tensors()

    return vis


class TestCrops:
    dir_temp = os.getcwd()
    dir_final = '/'.join(dir_temp.split('/')[0:-1])

    path = dir_final + '/images/SIDD_Medium_Srgb'

    def test_64_5_5(self):
        vis = image_crops(path=self.path)

        assert vis.crops[0][0].shape == (64, 64, 3) and \
               vis.crops[-1][0].shape == vis.crops[-1][1].shape and \
               list(vis.dataset[0]['GT'].size()) == [3, 64, 64] and \
               vis.dataset[-1]['Noisy'].size() == vis.dataset[-1]['GT'].size() and \
               len(vis.crops) == 25

    def test_256_3_3(self):
        vis = image_crops(path=self.path, size=256, sample_x=3, sample_y=3)

        assert vis.crops[0][0].shape == (256, 256, 3) and \
               vis.crops[-1][0].shape == vis.crops[-1][1].shape and \
               list(vis.dataset[0]['GT'].size()) == [3, 256, 256] and \
               vis.dataset[-1]['Noisy'].size() == vis.dataset[-1]['GT'].size() and \
               len(vis.crops) == 9
