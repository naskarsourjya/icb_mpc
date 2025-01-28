import torch
import torch.nn as nn
class Regressor(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, device = 'auto'):
        super(Regressor, self).__init__()

        # set device
        self._set_device(device= device)

        # Build the neural network dynamically based on the number of layers
        layers = []

        # Hidden layers
        for i in range(len(hidden_layers)):
            # Input layer
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_layers[i]))
                layers.append(nn.Tanh())

            # middle layers
            else:
                layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))  # Predicts a single value

        # Combine all layers
        self.network = nn.Sequential(*layers)

        # end
        return None

    def forward(self, x):
        """
        Forward pass for NARX model.
        states: Tensor of states (batch_size, n_x*order)
        inputs: Tensor of inputs (batch_size, n_u*order)
        """
        #x = torch.cat([states, inputs], dim=1)  # Concatenate input and output lags

        # returns current state
        return self.network(x)

    def count_parameters(self):
        """
        Count the total number of weights and biases in the network.
        Returns:
            int: Total number of parameters.
        """
        return sum(p.numel() for p in self.network.parameters())

    def _set_device(self, device):

        # auto choose gpu if gpu is available
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # torch default device is set
        self.torch_device = torch.device(device)
        torch.set_default_device(self.torch_device)

        # end
        return None


class MergedModel(nn.Module):
    def __init__(self, models, device = 'auto'):
        """
        Initializes the MergedModel with a list of models.

        Args:
            models (list of nn.Module): List of PyTorch models to merge.
        """
        super(MergedModel, self).__init__()

        # set device
        self._set_device(device=device)

        # storage
        self.model_list = models
        self.models = nn.ModuleList(models)

        # end
        return None

    def forward(self, x):
        """
        Forward pass through all models, combining their outputs.

        Args:
            x (torch.Tensor): Input tensor to be fed into all models.

        Returns:
            torch.Tensor: Concatenated outputs from all models.
        """
        # listing all outputs
        outputs = [model(x) for model in self.models]

        # end
        return torch.cat(outputs, dim=1)

    def _set_device(self, device):
        # auto choose gpu if gpu is available
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # torch default device is set
        self.torch_device = torch.device(device)
        torch.set_default_device(self.torch_device)

        # end
        return None