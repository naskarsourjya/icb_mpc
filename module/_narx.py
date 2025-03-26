import torch
import numpy as np
from torch.onnx.symbolic_opset9 import tensor
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ._neuralnetwork import Regressor



class narx():
    def __init__(self, n_x, n_u, order, t_step, set_seed = None, device = 'auto', dtype = torch.float64):

        if set_seed is not None:
            np.random.seed(set_seed)

        self.n_x = n_x
        self.n_u = n_u
        self.order = order
        self.t_step = t_step
        self.device = device
        self.set_seed = set_seed
        self._inputs = None
        self._states = None
        self.history = None
        self.dtype = dtype

        # default trainer settings
        self.setup_trainer()

        # setting up default plots
        self.setup_plot()

        # generating flags
        self.flags = {
            'narx_ready': False,
            'initial_condition_ready': False
        }

        pass


    def _set_device(self, torch_device):
        """
        Set the default device for the NARX model.

        Parameters
        ----------
        torch_device : str
            Device to be used for computation. Can be 'cpu' or 'cuda'.
        
        Returns
        -------
        None
        """

        torch.set_default_device(torch_device)

        return None


    def setup_trainer(self, hidden_layers=[50, 50, 50], batch_size=320, learning_rate=0.01, epochs= 1000,
                     validation_split = 0.2, scheduler_flag = True, lr_threshold = 1e-8):

        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.validation_split = validation_split
        self.scheduler_flag = scheduler_flag
        self.lr_threshold = lr_threshold



    def train(self, x_train, y_train):
        # init
        train_history = {'training_loss': [],
                         'validation_loss': [],
                         'learning_rate': [],
                         'epochs': []}

        # scaling the input
        scaler = StandardScaler()
        scaler.fit(x_train)

        # nn init
        narx_model = Regressor(input_size=self.order * (self.n_x + self.n_u),
                               output_size=self.n_x, hidden_layers=self.hidden_layers,
                               scaler=scaler, device=self.device)

        # setting up Mean Squared Error as loss function for training
        criterion = torch.nn.MSELoss()

        # setting up optimiser for training
        optimizer = torch.optim.AdamW(narx_model.parameters(), lr=self.learning_rate)

        # scheduler setup
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        # setting computation device
        self._set_device(torch_device= narx_model.torch_device)

        # converting datasets to tensors
        X_torch = torch.tensor(x_train.to_numpy(), dtype=self.dtype)
        Y_torch = torch.tensor(y_train.to_numpy(), dtype=self.dtype)

        # Create TensorDataset
        dataset = torch.utils.data.TensorDataset(X_torch, Y_torch)

        # splitting full datset
        train_dataset, validation_dataset = (
            torch.utils.data.random_split(dataset= dataset, lengths=[1-self.validation_split, self.validation_split],
                            generator=torch.Generator(device=narx_model.torch_device).manual_seed(self.set_seed)))

        # creating DataLoader with batch_size
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                            generator= torch.Generator(device=narx_model.torch_device).manual_seed(self.set_seed))
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset,
                                                            batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator(device=narx_model.torch_device).manual_seed(self.set_seed))

        # main training loop
        for epoch in tqdm(range(self.epochs), desc= 'Training NARX'):

            # narx training
            train_loss = 0
            for batch_X, batch_Y in train_dataloader:

                # Forward pass
                predictions = narx_model(batch_X).squeeze()
                loss = criterion(predictions, batch_Y)

                # Backward pass / parameters update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # storing loss
                train_loss += loss.item()

            # narx validation
            val_loss = 0
            for batch_X, batch_Y in validation_dataloader:
                with torch.no_grad():
                    predictions = narx_model(batch_X).squeeze()
                    val_loss += criterion(predictions, batch_Y).item()

            # storing data
            train_history['training_loss'].append(train_loss)
            train_history['validation_loss'].append(val_loss)
            train_history['epochs'].append(epoch)
            train_history['learning_rate'].append(optimizer.param_groups[0]["lr"])

            # learning rate update
            if self.scheduler_flag:
                lr_scheduler.step(val_loss)

                # break if training min learning rate is reached
                if optimizer.param_groups[0]["lr"] <= self.lr_threshold:
                    break

        # store model
        self.model = narx_model
        self.scaler = scaler
        self.train_history = train_history
        self.x_label = x_train.columns
        self.y_label = y_train.columns
        # flag update
        self.flags.update({
            'narx_ready': True,
        })

        # end
        return None


    def setup_plot(self, height_px=700, width_px=1800):

        self.height_px = height_px
        self.width_px = width_px

        return None

    def plot_narx_training_history_plotly(self):
        # Create subplots with secondary_y set in row 2
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=['Loss History', 'Learning Rate'],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]  # Enable secondary y-axis only in row 2
        )

        fig.update_layout(title_text='NARX Training History', height=self.height_px, width=self.width_px)

        # Extracting history
        train_history = self.train_history

        # Plot 1: Training Loss (primary y-axis in row 1)
        fig.add_trace(go.Scatter(x=train_history['epochs'], y=train_history['training_loss'],
                                 mode='lines', line=dict(color='green'),
                                 name=f'training loss',
                                 showlegend=True),
                      row=1, col=1)
        fig.update_yaxes(type='log', title_text='Training Loss', row=1, col=1)

        # Validation Loss (secondary y-axis in row 1)
        fig.add_trace(go.Scatter(x=train_history['epochs'], y=train_history['validation_loss'],
                                 mode='lines', line=dict(color='red'),
                                 name=f'validation loss',
                                 showlegend=True),
                      row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text='Validation Loss', type='log', row=1, col=1, secondary_y=True)
        fig.update_xaxes(title_text='epochs', row=1, col=1)

        # Plot 2: Learning Rate (row 2)
        fig.add_trace(go.Scatter(x=train_history['epochs'], y=train_history['learning_rate'],
                                 mode='lines', line=dict(color='blue'),
                                 showlegend=False),
                      row=2, col=1)
        fig.update_yaxes(type='log', title_text='Learning Rate', row=2, col=1)
        fig.update_xaxes(title_text='epochs', row=2, col=1)

        fig.show()

        # end
        return None