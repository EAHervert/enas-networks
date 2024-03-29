import unittest
import torch

from DNAS_DHDN import DIFFERENTIABLE_DHDN
from utilities.functions import generate_w_alphas


def test_weight_learning(k_value):
    loss = torch.nn.MSELoss()
    weights = generate_w_alphas(k_val=k_value)
    weights.requires_grad_(False)  # Training the model_weights

    DDHDN = DIFFERENTIABLE_DHDN.DifferentiableDHDN(k_value=k_value, channels=16)
    optimizer = torch.optim.SGD(DDHDN.parameters(), lr=0.01, momentum=0.9)

    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    y = torch.ones_like(x)

    with torch.no_grad():
        loss_old = loss(DDHDN(x), y)

    weights_prev = DDHDN.weights
    for i in range(10):
        optimizer.zero_grad()
        y_hat = DDHDN(x)
        loss_hat = loss(y_hat, y)
        loss_hat.backward()
        optimizer.step()

    weights_new = DDHDN.weights

    with torch.no_grad():
        loss_new = loss(DDHDN(x), y)

    boolean = list(weights_new) == list(weights_prev) and loss_new.item() < loss_old.item()

    return boolean, DDHDN


# Todo: Fix test_alpha_learning
def test_alpha_learning(k_value, model):
    loss = torch.nn.MSELoss()
    weights = generate_w_alphas(k_val=k_value)
    weights.requires_grad_(True)  # Training the architecture parameters

    optimizer = torch.optim.Adam([weights], lr=0.01)

    x = torch.randn(1, 3, 64, 64)
    y = torch.ones_like(x)

    with torch.no_grad():
        loss_old = loss(model(x), y)

    weights_prev = model.alphas
    for i in range(10):
        optimizer.zero_grad()
        model._update_w_alphas(weights)
        y_hat = model(x)
        print(y_hat.grad)
        loss_hat = loss(y_hat, y)
        loss_hat.backward()
        print(y_hat.grad)
        optimizer.step()

    weights_new = model.alphas

    with torch.no_grad():
        loss_new = loss(model(x), y)

    print(loss_old.item(), loss_new.item())
    print(weights_prev, weights_new)
    boolean = list(weights_new) != list(weights_prev)

    return boolean


def test_alphas(alphas, k_value):
    # Note we can initialize the weights to be alphas
    DDHDN = DIFFERENTIABLE_DHDN.DifferentiableDHDN(k_value=k_value, channels=16, weights=alphas)
    x = torch.randn(1, 3, 64, 64, requires_grad=True)

    y = DDHDN(x)

    return list(x.shape) == list(y.shape)


class test_DIFFERENTIABLE_DHDN(unittest.TestCase):
    def test_architecture(self):
        assert test_alphas(alphas=None, k_value=1)
        assert test_alphas(alphas=None, k_value=2)
        assert test_alphas(alphas=None, k_value=3)
        bool_weight, model = test_weight_learning(k_value=1)
        assert bool_weight
        assert test_alpha_learning(k_value=1, model=model)


if __name__ == '__main__':
    unittest.main()
