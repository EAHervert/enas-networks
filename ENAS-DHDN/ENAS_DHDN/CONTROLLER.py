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
from utilities.functions import macro_array, reduced_macro_array


class Controller(nn.Module):
    """
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py
    """

    def __init__(self,
                 k_value=3,
                 kernel_bool=True,
                 down_bool=True,
                 up_bool=True,
                 lstm_size=32,
                 lstm_num_layers=1):
        super(Controller, self).__init__()

        # Network Hyperparameters
        self.k_value = k_value

        # Booleans
        self.kernel_bool = kernel_bool
        self.down_bool = down_bool
        self.up_bool = up_bool

        # Arrays
        self.kernel_array = [0 for _ in range(4 * self.k_value + 2)]
        self.down_array = [0 for _ in range(self.k_value)]
        self.up_array = [0 for _ in range(self.k_value)]
        self.sample_arc = macro_array(self.k_value, self.kernel_array, self.down_array, self.up_array)

        # Values for Controller Training
        self.sample_entropy = None
        self.sample_log_prob = None

        # LSTM Network Parameters
        # The controller will contain LSTM cells.
        self.LSTM_size = lstm_size
        self.LSTM_num_layers = lstm_num_layers

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

        if self.kernel_bool:
            # Will take the output of the LSTM and give values corresponding to the DCR Blocks
            self.w_kernel = nn.Linear(self.LSTM_size, 8, bias=False)
            self.w_emb_kernel = nn.Embedding(8, self.LSTM_size)
        else:
            self.w_kernel = None
            self.w_emb_kernel = None

        if self.down_bool:
            # Will take the output of the LSTM and give values corresponding to the Down Blocks
            self.w_down = nn.Linear(self.LSTM_size, 3, bias=False)
            self.w_emb_down = nn.Embedding(3, self.LSTM_size)
        else:
            self.w_down = None
            self.w_emb_down = None

        if self.up_bool:
            # Will take the output of the LSTM and give values corresponding to the Down Blocks
            self.w_up = nn.Linear(self.LSTM_size, 3, bias=False)
            self.w_emb_up = nn.Embedding(3, self.LSTM_size)
        else:
            self.w_up = None
            self.w_emb_up = None

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

        # For the REINFORCE Algorithm.
        entropies = []
        log_probs = []

        inputs = self.g_emb.weight
        w_array = [self.w_kernel, self.w_down, self.w_up]
        arrays = [self.kernel_array, self.down_array, self.up_array]
        emb = [self.w_emb_kernel, self.w_emb_down, self.w_emb_up]
        indexes = [-1, -1, -1]
        # Generate the Architecture:
        for i in range(6 * self.k_value + 2):
            if (i + 1) % 3 != 0:
                index = 0
                boolean = self.kernel_bool
            elif i < 3 * self.k_value + 1:
                index = 1
                boolean = self.down_bool
            else:
                index = 2
                boolean = self.up_bool

            indexes[index] += 1
            if boolean:
                inputs = inputs.unsqueeze(0)  # Will return a tensor with dimension 1 by dim(inputs).

                # Feed in the input tensor which specifies the input and the hidden state from the previous step
                output, hn = self.w_lstm(inputs, h0)
                output = output.squeeze(0)  # Will return a tensor with dimension dim(inputs)[original].
                h0 = hn  # Have the hidden output be the initial hidden input for the next step.

                logit = w_array[index](output)  # Using the output and passing it through a linear layer.
                # Have the network generate probabilities to pick the layers that we need.
                probs = self.softmax(logit)
                DCR_dist = Categorical(probs=probs)
                DCR_layer = DCR_dist.sample()

                arrays[index][indexes[index]] = DCR_layer.item()

                # Here we have the log probabilities and entropy of the distribution.
                # These values will be used for the REINFORCE Algorithm
                log_prob = DCR_dist.log_prob(DCR_layer)
                log_probs.append(log_prob.view(-1))
                entropy = DCR_dist.entropy()
                entropies.append(entropy.view(-1))

                inputs = emb[index](DCR_layer)

                del output, hn, logit, probs, log_prob, entropy

        self.sample_arc = macro_array(self.k_value, self.kernel_array, self.down_array, self.up_array)

        self.sample_entropy = torch.sum(torch.cat(entropies))
        self.sample_log_prob = torch.sum(torch.cat(log_probs))


# Reduced Controller will work with just one layer of the encoder and one layer of the decoder. It will stack these
# layers k_value times.
class ReducedController(nn.Module):
    def __init__(self,
                 k_value=3,
                 encoder=True,
                 bottleneck=True,
                 decoder=True,
                 lstm_size=32,
                 lstm_num_layers=1):
        super(ReducedController, self).__init__()

        # Network Hyperparameters
        self.k_value = k_value

        # Booleans
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder

        # Arrays
        self.encoder_array = [0, 0, 0]
        self.bottleneck_array = [0, 0]
        self.decoder_array = [0, 0, 0]

        # Values for Controller Training
        self.sample_entropy = None
        self.sample_log_prob = None

        # LSTM Network Parameters
        # The controller will contain LSTM cells.
        self.LSTM_size = lstm_size
        self.LSTM_num_layers = lstm_num_layers

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

        if self.encoder or self.bottleneck or self.decoder:
            # Will take the output of the LSTM and give values corresponding to the DCR Blocks
            self.w_kernel = nn.Linear(self.LSTM_size, 8, bias=False)
            self.w_emb_kernel = nn.Embedding(8, self.LSTM_size)
        else:
            self.w_kernel = None
            self.w_emb_kernel = None

        if self.encoder:
            # Will take the output of the LSTM and give values corresponding to the Down Blocks
            self.w_down = nn.Linear(self.LSTM_size, 3, bias=False)
            self.w_emb_down = nn.Embedding(3, self.LSTM_size)
        else:
            self.w_down = None
            self.w_emb_down = None

        if self.decoder:
            # Will take the output of the LSTM and give values corresponding to the Down Blocks
            self.w_up = nn.Linear(self.LSTM_size, 3, bias=False)
            self.w_emb_up = nn.Embedding(3, self.LSTM_size)
        else:
            self.w_up = None
            self.w_emb_up = None

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

        # For the REINFORCE Algorithm.
        entropies = []
        log_probs = []

        inputs = self.g_emb.weight
        w_array = [self.w_kernel, self.w_down, self.w_up]
        emb = [self.w_emb_kernel, self.w_emb_down, self.w_emb_up]
        # Generate the Architecture:
        # Encoder
        for i in range(3):
            if i == 2:
                index = 1
            else:
                index = 0
            inputs = inputs.unsqueeze(0)  # Will return a tensor with dimension 1 by dim(inputs).

            # Feed in the input tensor which specifies the input and the hidden state from the previous step
            output, hn = self.w_lstm(inputs, h0)
            output = output.squeeze(0)  # Will return a tensor with dimension dim(inputs)[original].
            h0 = hn  # Have the hidden output be the initial hidden input for the next step.

            logit = w_array[index](output)  # Using the output and passing it through a linear layer.
            # Have the network generate probabilities to pick the layers that we need.
            probs = self.softmax(logit)
            DCR_dist = Categorical(probs=probs)
            DCR_layer = DCR_dist.sample()

            self.encoder_array[i] = DCR_layer.item()

            # Here we have the log probabilities and entropy of the distribution.
            # These values will be used for the REINFORCE Algorithm
            log_prob = DCR_dist.log_prob(DCR_layer)
            log_probs.append(log_prob.view(-1))
            entropy = DCR_dist.entropy()
            entropies.append(entropy.view(-1))

            inputs = emb[index](DCR_layer)

            del output, hn, logit, probs, log_prob, entropy

        # Bottlenect
        for i in range(2):
            index = 0
            inputs = inputs.unsqueeze(0)  # Will return a tensor with dimension 1 by dim(inputs).

            # Feed in the input tensor which specifies the input and the hidden state from the previous step
            output, hn = self.w_lstm(inputs, h0)
            output = output.squeeze(0)  # Will return a tensor with dimension dim(inputs)[original].
            h0 = hn  # Have the hidden output be the initial hidden input for the next step.

            logit = w_array[index](output)  # Using the output and passing it through a linear layer.
            # Have the network generate probabilities to pick the layers that we need.
            probs = self.softmax(logit)
            DCR_dist = Categorical(probs=probs)
            DCR_layer = DCR_dist.sample()

            self.bottleneck_array[i] = DCR_layer.item()

            # Here we have the log probabilities and entropy of the distribution.
            # These values will be used for the REINFORCE Algorithm
            log_prob = DCR_dist.log_prob(DCR_layer)
            log_probs.append(log_prob.view(-1))
            entropy = DCR_dist.entropy()
            entropies.append(entropy.view(-1))

            inputs = emb[index](DCR_layer)

            del output, hn, logit, probs, log_prob, entropy

        # Decoder
        for i in range(3):
            if i == 0:
                index = 2
            else:
                index = 0
            inputs = inputs.unsqueeze(0)  # Will return a tensor with dimension 1 by dim(inputs).

            # Feed in the input tensor which specifies the input and the hidden state from the previous step
            output, hn = self.w_lstm(inputs, h0)
            output = output.squeeze(0)  # Will return a tensor with dimension dim(inputs)[original].
            h0 = hn  # Have the hidden output be the initial hidden input for the next step.

            logit = w_array[index](output)  # Using the output and passing it through a linear layer.
            # Have the network generate probabilities to pick the layers that we need.
            probs = self.softmax(logit)
            DCR_dist = Categorical(probs=probs)
            DCR_layer = DCR_dist.sample()

            self.decoder_array[i] = DCR_layer.item()

            # Here we have the log probabilities and entropy of the distribution.
            # These values will be used for the REINFORCE Algorithm
            log_prob = DCR_dist.log_prob(DCR_layer)
            log_probs.append(log_prob.view(-1))
            entropy = DCR_dist.entropy()
            entropies.append(entropy.view(-1))

            inputs = emb[index](DCR_layer)

            del output, hn, logit, probs, log_prob, entropy

        self.sample_arc = reduced_macro_array(k=self.k_value,
                                              encoder_array=self.encoder_array,
                                              bottleneck_array=self.bottleneck_array,
                                              decoder_array=self.decoder_array)

        self.sample_entropy = torch.sum(torch.cat(entropies))
        self.sample_log_prob = torch.sum(torch.cat(log_probs))
