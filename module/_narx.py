import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
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
        input_scaler = StandardScaler()
        input_scaler.fit(x_train)

        output_scaler = StandardScaler()
        output_scaler.fit(y_train)

        # nn init
        narx_model = Regressor(input_size=self.order * (self.n_x + self.n_u),
                               output_size=self.n_x, hidden_layers=self.hidden_layers,
                               input_scaler=input_scaler, output_scaler=output_scaler, device=self.device)

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
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.train_history = train_history
        self.x_label = x_train.columns
        self.y_label = y_train.columns
        # flag update
        self.flags.update({
            'narx_ready': True,
        })

        # end
        return None


    def setup_plot(self, height_px=9, width_px=16):

        self.height_px = height_px
        self.width_px = width_px

        return None


    def plot_narx_training_history(self):
        train_history = self.train_history

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.width_px, self.height_px), sharex=True)

        fig.suptitle("NARX Training History")

        # Plot 1: Training and Validation Loss with dual y-axis
        color1 = 'green'
        ax1.plot(train_history['epochs'], train_history['training_loss'], color=color1, label='Training Loss')
        ax1.set_ylabel("Training Loss", color=color1)
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor=color1)

        ax1b = ax1.twinx()  # Secondary y-axis
        color2 = 'red'
        ax1b.plot(train_history['epochs'], train_history['validation_loss'], color=color2, label='Validation Loss')
        ax1b.set_ylabel("Validation Loss", color=color2)
        ax1b.set_yscale('log')
        ax1b.tick_params(axis='y', labelcolor=color2)

        # Legends
        ax1.legend(loc='upper left')
        ax1b.legend(loc='upper right')
        ax1.set_title('Loss History')

        # Plot 2: Learning Rate
        ax2.plot(train_history['epochs'], train_history['learning_rate'], color='blue')
        ax2.set_ylabel("Learning Rate")
        ax2.set_xlabel("Epochs")
        ax2.set_yscale('log')
        ax2.set_title('Learning Rate')

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
        plt.show()

        return None
