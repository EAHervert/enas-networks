import unittest
import torch

from DNAS_DHDN import DIFFERENTIABLE_DHDN
from utilities.functions import generate_w_alphas


def test_weight_learning(k_value):
    loss = torch.nn.MSELoss()
    weights = generate_w_alphas(k_val=k_value)
    weights.requires_grad_(False)  # Training the model_weights

    DDHDN = DIFFERENTIABLE_DHDN.DifferentiableDHDN(k_value=k_value, channels=16)
    optimizer = torch.optim.SGD(DDHDN.parameters(), lr=0.001, momentum=0.9)

    x = torch.randn(1, 3, 64, 64)
    y = torch.ones_like(x)

    with torch.no_grad():
        loss_old = loss(DDHDN(x, weights=weights), y)

    for i in range(100):
        optimizer.zero_grad()
        y_hat = DDHDN(x, weights=weights)
        loss_hat = loss(y_hat, y)
        loss_hat.backward()
        optimizer.step()

    with torch.no_grad():
        loss_new = loss(DDHDN(x, weights=weights), y)

    boolean = loss_new.item() < loss_old.item()

    return boolean, DDHDN


def test_weight_alpha_learning(k_value, model):
    loss = torch.nn.MSELoss()
    weights = generate_w_alphas(k_val=k_value, s_val=1)
    weights.requires_grad_(True)  # Training the architecture parameters

    optimizer = torch.optim.Adam([weights], lr=0.01)
    x = torch.randn(1, 3, 64, 64)
    y = torch.ones_like(x)

    with torch.no_grad():
        loss_old = loss(model(x, weights), y)

    weights_prev = weights.clone().detach()
    for i in range(100):
        optimizer.zero_grad()
        y_hat = model(x, weights)
        loss_hat = loss(y_hat, y)
        loss_hat.backward()
        optimizer.step()

    weights_new = weights.clone().detach()

    with torch.no_grad():
        loss_new = loss(model(x, weights), y)

    boolean = list(weights_new) != list(weights_prev) and loss_new.item() < loss_old.item()

    return boolean


def test_weights(weights, k_value):
    if weights is None:
        weights = generate_w_alphas(k_val=k_value)
    # Note we can initialize the weights to be alphas
    DDHDN = DIFFERENTIABLE_DHDN.DifferentiableDHDN(k_value=k_value, channels=16)
    x = torch.randn(1, 3, 64, 64, requires_grad=True)

    y = DDHDN(x, weights=weights)

    return list(x.shape) == list(y.shape)


class test_DIFFERENTIABLE_DHDN(unittest.TestCase):
    def test_architecture(self):
        assert test_weights(weights=None, k_value=1)
        assert test_weights(weights=None, k_value=2)
        assert test_weights(weights=None, k_value=3)
        # Test: k_val = 1
        bool_weight, model = test_weight_learning(k_value=1)
        assert bool_weight
        assert test_weight_alpha_learning(k_value=1, model=model)

        # Test: k_val = 2
        bool_weight, model = test_weight_learning(k_value=2)
        assert bool_weight
        assert test_weight_alpha_learning(k_value=2, model=model)

        # Test: k_val = 3
        bool_weight, model = test_weight_learning(k_value=3)
        assert bool_weight
        assert test_weight_alpha_learning(k_value=3, model=model)


if __name__ == '__main__':
    unittest.main()
