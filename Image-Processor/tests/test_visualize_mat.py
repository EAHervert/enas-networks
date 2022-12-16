from Image_Processor import visualize_mat
import numpy as np

mat_class = visualize_mat.Validation()


def model(input):
    return input


mat_class.evaluate_model(model, True)
