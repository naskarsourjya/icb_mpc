import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


class Regressor(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, scaler=None, device='auto'):
        super(Regressor, self).__init__()

        # set device
        self._set_device(device=device)

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

        # stores number of parameters
        self.n_params = self.count_parameters()

        # setup scalar
        if scaler != None:
            self.setup_scaler(scaler)
            self.scaler_flag = True

        else:
            self.scaler_flag = False

        # end of init


    def setup_scaler(self, scaler):

        assert isinstance(scaler, MinMaxScaler), (f"Only MinMaxScaler supported as of yet! "
                                                  f"Scaler found is {scaler}, which is not supported.")

        # extracting scaler info
        X_min = torch.tensor(scaler.data_min_, dtype=torch.float32)  # Minimum values of original data
        X_max = torch.tensor(scaler.data_max_, dtype=torch.float32)  # Maximum values of original data
        X_scale = torch.tensor(scaler.scale_, dtype=torch.float32)  # Scaling factor (1 / (X_max - X_min))
        X_min_target = torch.tensor(scaler.min_, dtype=torch.float32)  # Shift factor (used for transformation)

        # storage
        self.scaler = {'scaler': scaler,
                       'X_min': X_min,
                       'X_max': X_max,
                       'X_scale': X_scale,
                       'X_min_target': X_min_target}

        # end
        return None


    def forward(self, x):

        # init
        X_min_target = self.scaler['X_min_target']
        X_scale = self.scaler['X_scale']
        X_min = self.scaler['X_min']

        # scaling
        x_scaled = X_min_target + X_scale * (x - X_min)

        # returns current state
        return self.network(x_scaled)


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

            # 1st priority
            if torch.cuda.is_available():
                device = 'cuda'

            # 2nd priority
            #elif torch.backends.mps.is_available():
            #    device = 'mps'

            # fallback
            else:
                device = 'cpu'

        # torch default device is set
        self.torch_device = torch.device(device)
        torch.set_default_device(self.torch_device)

        # end
        return None


class MergedModel(nn.Module):
    def __init__(self, models, device='auto'):
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

        self.n_params = 0
        for model in models:
            self.n_params += model.n_params

        # end of init


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

            # 1st priority
            if torch.cuda.is_available():
                device = 'cuda'

            # 2nd priority
            #elif torch.backends.mps.is_available():
            #    device = 'mps'

            # fallback
            else:
                device = 'cpu'

        # torch default device is set
        self.torch_device = torch.device(device)
        torch.set_default_device(self.torch_device)

        # end
        return None
