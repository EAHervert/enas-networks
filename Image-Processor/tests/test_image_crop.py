from Image_Processor import image_crop

vis = image_crop.Crops()

vis.set_random_images()
vis.get_random_crops()
vis.set_tensors()
vis.plot_tensors()
