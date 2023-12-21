"""
The following controller will be generating a Macro-Array specifying an architecture based on three smaller
Micro-Arrays. The parameter k will specify the overall size of the network, which is 2k + 1.

The first array is an array of the kernels in the DCR Block. This array is of size 4k + 2. The second array is an
array specifying which upsampling is performed and the third array is an array specifying which downsampling is
performed. These arrays are of size k

The default arrays are arrays of size 0
The zero Macro-Array represents the original DHDN Network.
"""

# Libraries that will be used.
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from utilities.functions import macro_array


class Controller(nn.Module):
    """
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py
    """

    def __init__(self,
                 k_value=3,
                 kernel_bool=True,
                 down_bool=True,
                 up_bool=True,
                 LSTM_size=32,
                 LSTM_num_layers=1):
        super(Controller, self).__init__()

        # Network Hyperparameters
        self.k_value = k_value

        # Booleans
        self.kernel_bool = kernel_bool
        self.down_bool = down_bool
        self.up_bool = up_bool

        # Arrays
        self.kernel_array = []
        self.down_array = []
        self.up_array = []

        # LSTM Network Parameters
        # The controller will contain LSTM cells.
        self.LSTM_size = LSTM_size
        self.LSTM_num_layers = LSTM_num_layers

        self._create_params()

    # Here we create the parameters of the Controller.
    def _create_params(self):
        """
        https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L83

        This is the lstm portion of the network. We have that the input and hidden state will be both of size
        "lstm_size". We will have that this will be a stacked LSTM where the number of layers stacked is given
        "lstm_num_layers". The stacked LSTM will take outputs of previous LSTM cells and use them as inputs.
        """
        self.w_lstm = nn.LSTM(input_size=self.LSTM_size,
                              hidden_size=self.LSTM_size,
                              num_layers=self.LSTM_num_layers)

        # https://pytorch.org/docs/master/generated/torch.nn.Embedding.html
        self.g_emb = nn.Embedding(1, self.LSTM_size)

        # The layer outputs embedded into the LSTM_Size
        self.w_emb_kernel = nn.Embedding(8, self.LSTM_size)
        self.w_emb_down = nn.Embedding(3, self.LSTM_size)
        self.w_emb_up = nn.Embedding(3, self.LSTM_size)

        if self.kernel_bool:
            # Will take the output of the LSTM and give values corresponding to the DCR Blocks
            self.w_kernel = nn.Linear(self.LSTM_size, 8, bias=False)

        if self.down_bool:
            # Will take the output of the LSTM and give values corresponding to the Down Blocks
            self.w_down = nn.Linear(self.LSTM_size, 3, bias=False)

        if self.up_bool:
            # Will take the output of the LSTM and give values corresponding to the Down Blocks
            self.w_up = nn.Linear(self.LSTM_size, 3, bias=False)

        self.softmax = nn.Softmax(dim=1)

        self._reset_params()

    def _reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)  # Will make the weights all in the range (-0.1, 0.1)

        # Initializing weights to be in (-0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)

    def forward(self):
        """
        https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L126
        """
        # Since we have that LSTMs require two inputs at each time step, x_i and h_i, we have that at time t = 0,
        # the input for the hidden state will be h_0 = vector(0)
        h0 = None  # setting h0 to None will initialize LSTM state with 0s

        # Reinitialize the arrays:
        self.kernel_array = []
        self.down_array = []
        self.up_array = []

        # For the REINFORCE Algorithm.
        entropies = []
        log_probs = []

        inputs = self.g_emb.weight

        # DCR Blocks:
        if self.kernel_bool:

            # Let us do the Kernel Array:
            for DCR_block in range(4 * self.k_value + 2):
                inputs = inputs.unsqueeze(0)  # Will return a tensor with dimension 1 by dim(inputs).

                # Feed in the input tensor which specifies the input and the hidden state from the previous step
                output, hn = self.w_lstm(inputs, h0)
                output = output.squeeze(0)  # Will return a tensor with dimension dim(inputs)[original].
                h0 = hn  # Have the hidden output be the initial hidden input for the next step.

                logit = self.w_kernel(output)  # Using the output and passing it through a linear layer.
                # Have the network generate probabilities to pick the layers that we need.
                probs = self.softmax(logit)
                DCR_dist = Categorical(probs=probs)
                DCR_layer = DCR_dist.sample()

                self.kernel_array.append(DCR_layer.item())

                # Here we have the log probabilities and entropy of the distribution.
                # These values will be used for the REINFORCE Algorithm
                log_prob = DCR_dist.log_prob(DCR_layer)
                log_probs.append(log_prob.view(-1))
                entropy = DCR_dist.entropy()
                entropies.append(entropy.view(-1))

                inputs = self.w_emb_kernel(DCR_layer)

        else:
            self.kernel_array = [0] * (4 * self.k_value + 2)

        # Down Blocks:
        if self.down_bool:

            # Let us do the Down Array:
            for down_block in range(self.k_value):
                inputs = inputs.unsqueeze(0)  # Will return a tensor with dimension 1xdim(inputs).

                # Feed in the input tensor which specifies the input and the hidden state from the previous step
                output, hn = self.w_lstm(inputs, h0)
                output = output.squeeze(0)  # Will return a tensor with dimension dim(inputs)[original].
                h0 = hn  # Have the hidden output be the initial hidden input for the next step.

                # Since we are generating the Down Blocks:
                logit = self.w_down(output)  # Using the output and passing it through a linear layer.
                # Have the network generate probabilities to pick the layers that we need.
                probs = self.softmax(logit)
                down_dist = Categorical(probs=probs)
                down_layer = down_dist.sample()

                # Append the
                self.down_array.append(down_layer.item())

                # Here we have the log probabilities and entropy of the distribution.
                # These values will be used for the REINFORCE Algorithm
                log_prob = down_dist.log_prob(down_layer)
                log_probs.append(log_prob.view(-1))
                entropy = down_dist.entropy()
                entropies.append(entropy.view(-1))

                inputs = self.w_emb_down(down_layer)

        else:
            self.down_array = [0] * (4 * self.k_value + 2)

        # Up Blocks:
        if self.up_bool:

            # Let us do the Up Array:
            for DCR_block in range(self.k_value):
                inputs = inputs.unsqueeze(0)  # Will return a tensor with dimension 1xdim(inputs).

                # Feed in the input tensor which specifies the input and the hidden state from the previous step
                output, hn = self.w_lstm(inputs, h0)
                output = output.squeeze(0)  # Will return a tensor with dimension dim(inputs)[original].
                h0 = hn  # Have the hidden output be the initial hidden input for the next step.

                # Since we are generating the Up Blocks:
                logit = self.w_up(output)  # Using the output and passing it through a linear layer.
                # Have the network generate probabilities to pick the layers that we need.
                probs = self.softmax(logit)
                up_dist = Categorical(probs=probs)
                up_layer = up_dist.sample()

                # Append the
                self.up_array.append(up_layer.item())

                # Here we have the log probabilities and entropy of the distripution.
                # These values will be used for the REINFORCE Algorithm
                log_prob = up_dist.log_prob(up_layer)
                log_probs.append(log_prob.view(-1))
                entropy = up_dist.entropy()
                entropies.append(entropy.view(-1))

                inputs = self.w_emb_up(up_layer)

        else:
            self.up_array = [0] * (4 * self.k_value + 2)

        self.sample_arc = macro_array(self.k_value, self.kernel_array, self.down_array, self.up_array)

        entropies = torch.cat(entropies)
        self.sample_entropy = torch.sum(entropies)

        log_probs = torch.cat(log_probs)
        self.sample_log_prob = torch.sum(log_probs)
