from Image_Processor import visualize_mat
import numpy as np
import torch

mat_class = visualize_mat.Validation()


def model(input_, alpha=0.01):
    noise = alpha * torch.randn(input_.size())
    out_ = input_ + noise

    return (out_ - torch.min(out_)) / (torch.max(out_) - torch.min(out_))


mat_class.visdom_client_setup()
mat_class.evaluate_model(model, True, index=0)
mat_class.evaluate_model(model, True, index=1)
