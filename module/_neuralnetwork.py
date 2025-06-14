import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Regressor(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, input_scaler=None, output_scaler=None, device='auto',
                 dtype = torch.float64):
        super(Regressor, self).__init__()

        # Set device
        self._set_device(device=device, dtype=dtype)
        self.dtype = dtype

        # Build the neural network dynamically based on the number of layers
        layers = []

        # Input layer (from input_size to the first hidden layer)
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.Tanh())

        # Hidden layers (from one hidden layer to the next)
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(nn.Tanh())

        # Output layer (from the last hidden layer to output_size)
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers).to(self.torch_device)  # Move network to device

        # Store number of parameters
        self.n_params = self.count_parameters()

        # Setup input scaler
        if input_scaler is not None:
            self.input_scaler = self.setup_scaler(input_scaler)
            self.input_scaler_flag = True
        else:
            self.input_scaler_flag = False

        # setup output scaler
        if output_scaler is not None:
            self.output_scaler = self.setup_scaler(output_scaler)
            self.output_scaler_flag = True
        else:
            self.output_scaler_flag = False


    def setup_scaler(self, scaler):
        if isinstance(scaler, MinMaxScaler):

            # Extract scaler info and move tensors to the correct device
            X_min = torch.tensor(scaler.data_min_, dtype=self.dtype).to(self.torch_device)  # Minimum values
            X_max = torch.tensor(scaler.data_max_, dtype=self.dtype).to(self.torch_device)  # Maximum values
            X_scale = torch.tensor(scaler.scale_, dtype=self.dtype).to(self.torch_device)  # Scaling factor
            X_min_target = torch.tensor(scaler.min_, dtype=self.dtype).to(self.torch_device)  # Shift factor

            # Store scaler info
            scaler = {
                'scaler': scaler,
                'X_min': X_min,
                'X_max': X_max,
                'X_scale': X_scale,
                'X_min_target': X_min_target
            }

        elif isinstance(scaler, StandardScaler):

            # Extract scaler info and move tensors to the correct device
            mean = torch.tensor(scaler.mean_, dtype=self.dtype).to(self.torch_device)  # Mean values
            std = torch.tensor(scaler.scale_, dtype=self.dtype).to(self.torch_device)  # Standard deviations

            # Store scaler info
            scaler = {
                'scaler': scaler,
                'mean': mean,
                'std': std
            }

        else:
            raise ValueError(f"Only MinMaxScaler or StandardScaler supported as of yet! "
                             f"Scaler found is {scaler}, which is not supported.")

        return scaler


    def scale_input_layer(self, scaler, x_unscaled):

        if isinstance(scaler['scaler'], MinMaxScaler):
            X_min_target = scaler['X_min_target']
            X_scale = scaler['X_scale']
            X_min = scaler['X_min']
            x_scaled = X_min_target + X_scale * (x_unscaled - X_min)

        elif isinstance(scaler['scaler'], StandardScaler):
            mean = scaler['mean']
            std = scaler['std']
            x_scaled = (x_unscaled - mean) / std

        else:
            raise ValueError(f"Only MinMaxScaler or StandardScaler supported as of yet! "
                             f"Scaler found is {scaler}, which is not supported.")

        return x_scaled


    def unscale_output_layer(self, scaler, x_scaled):

        if isinstance(scaler['scaler'], MinMaxScaler):
            X_min_target = scaler['X_min_target']
            X_scale = scaler['X_scale']
            X_min = scaler['X_min']
            x_unscaled = (x_scaled - X_min_target)/X_scale + X_min

        elif isinstance(scaler['scaler'], StandardScaler):
            mean = scaler['mean']
            std = scaler['std']
            x_unscaled = x_scaled * std + mean

        else:
            raise ValueError(f"Only MinMaxScaler or StandardScaler supported as of yet! "
                             f"Scaler found is {scaler}, which is not supported.")

        return x_unscaled

    def forward(self, x_scaled):

        #raise RuntimeError('Only for training!')

        return self.network(x_scaled)


    def evaluate(self, x):
        # Ensure input tensor is on the correct device
        x = x.to(self.torch_device)

        # scaling input layer
        if self.input_scaler_flag:
            x_scaled = self.scale_input_layer(scaler=self.input_scaler, x_unscaled=x)
        else:
            x_scaled = x

        # evaluating nn
        scaled_y = self.network(x_scaled)

        # scaling input layer
        if self.output_scaler_flag:
            y_output_unscaled = self.unscale_output_layer(scaler=self.output_scaler, x_scaled=scaled_y)
        else:
            y_output_unscaled = scaled_y

        # Forward pass through the network
        return y_output_unscaled


    def count_parameters(self):
        """Count the total number of weights and biases in the network."""
        return sum(p.numel() for p in self.network.parameters())


    def _set_device(self, device, dtype):
        # Auto choose GPU if available
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            #elif torch.backends.mps.is_available():
            #    device = 'mps'
            else:
                device = 'cpu'

        # Set device
        self.torch_device = torch.device(device)

        # set default data type
        torch.set_default_dtype(dtype)


class MergedModel(nn.Module):
    def __init__(self, models, device='auto', dtype=torch.float64):
        """
        Initializes the MergedModel with a list of models.

        Args:
            models (list of nn.Module): List of PyTorch models to merge.
        """
        super(MergedModel, self).__init__()

        # set device
        self._set_device(device=device, dtype=dtype)

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
        outputs = [model.evaluate(x) for model in self.models]

        # end
        return torch.cat(outputs, dim=1)


    def _set_device(self, device, dtype):
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

        # set default data type
        torch.set_default_dtype(dtype)

        # end
        return None
