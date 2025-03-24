import torch
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ._neuralnetwork import Regressor



class narx():
    def __init__(self, n_x, n_u, order, t_step, set_seed = None, device = 'auto'):

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


    def reshape(self, array, shape):
        """
        Reshape the array to the desired shape.

        Parameters
        ----------
        array : numpy.array
            Array to be reshaped.
        shape : tuple
            Desired shape of the array.

        Returns
        -------
        numpy.array
            Reshaped array.
        """

        # rows and columns
        rows, cols = shape

        # end
        return array.reshape(cols, rows).T


    @property
    def states(self):
        return self._states


    @states.setter
    def states(self, val):
        assert isinstance(val, np.ndarray), "states must be a numpy.array."

        assert val.shape[1] == self.order, \
            'Number of samples must be equal to the order of the NARX model!'

        assert val.shape[0] == self.n_x, (
            'Expected number of states is: {}, but found {}'.format(self.n_x, val.shape[0]))

        # storage
        self._states = val


    @property
    def inputs(self):
        return self._inputs


    @inputs.setter
    def inputs(self, val):
        if self.order > 1:
            assert isinstance(val, np.ndarray), "inputs must be a numpy.array."

            assert self.order - 1 == val.shape[1], \
                'Number of samples for inputs should be (order-1) !'

            assert val.shape[0] == self.n_u, (
                'Expected number of inputs is: {}, but found {}'.format(self.n_u, val.shape[0]))

            # storage
            self._inputs = val

        # error
        else:
            raise ValueError("Inputs cannot be set for system with order <= 1.")


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
        scaler.fit(x_train.T)

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
        X_torch = torch.tensor(x_train.T, dtype=torch.float32)
        Y_torch = torch.tensor(y_train.T, dtype=torch.float32)

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
        # flag update
        self.flags.update({
            'narx_ready': True,
        })

        # end
        return None


    def set_initial_guess(self):

        assert self.flags['narx_ready'] == True, \
            'NARX not trained! Train the NARX model.'

        assert self.flags['cqr_ready'] == True, \
            'QR not confromalised! Conformalise QR model.'

        states = self.states
        inputs = self.inputs

        state_order = self.order
        state_samples = states.shape[1]

        # stacking states and inputs with order
        order_states = np.vstack([states[:, state_order - i - 1:state_samples - i] for i in range(state_order)])

        # if order is 2 or more, only then previous inputs are needed
        if self.order > 1:
            input_order = self.order - 1
            input_samples = inputs.shape[1]

            # stacking layer
            order_inputs = np.vstack([inputs[:, input_order - i - 1:input_samples - i] for i in range(input_order)])

            # stacking states and inputs for narx model
            initial_cond = np.vstack([order_states, order_inputs])

        else:
            initial_cond = order_states

        # store cqr initial contition
        self.initial_cond = initial_cond

        # flag update
        self.flags.update({
            'initial_condition_ready': True
        })

        # end
        return None


    def make_step(self, u0):
        assert self.flags['narx_ready'], "NARX not trained."

        assert self.flags['initial_condition_ready'], "NARX not initialised"

        assert u0.shape[0] == self.n_u, \
            f"u0 should have have {self.n_u} rows but instead found {u0.shape[0]}!"

        assert u0.shape[1] == 1, \
            f"u0 should have have 1 columns but instead found {u0.shape[1]}!"

        # init
        x0 = self.initial_cond
        n_x = self.n_x
        n_u = self.n_u
        order = self.order


        # segregating states and inputs
        states = x0[0:n_x*order, :]
        inputs = x0[n_x*order:, :]

        # stacking all data
        X = np.vstack([states, u0, inputs])

        # setting default device
        self._set_device(torch_device=self.model.torch_device)

        # scaling
        #X_scaled = self.scaler.transform(X.T)

        # loading tensor
        X_torch = torch.tensor(X.T, dtype=torch.float32)

        # making full model prediction
        with torch.no_grad():
            y_pred = self.model(X_torch).cpu().numpy().T

        # reshaping from a column vector to row with states and column with different quantiles
        x0 = self.reshape(y_pred, shape=(n_x, -1))

        # pushing oldest state out of system and inserting the current state
        new_states = np.vstack([states[n_x:(order) * n_x, :], x0])

        # setting new initial conditions
        if order>1:

            # pushing oldest input out of system and inserting the current input
            new_inputs = np.vstack([inputs[n_u:(order - 1) * n_u, :], u0])

            # setting new initial guess by removing the last timestamp data
            self.states=self.reshape(new_states, shape=(n_x, -1))
            self.inputs=self.reshape(new_inputs, shape=(n_u, -1))
            self.set_initial_guess()

        else:
            self.states = self.reshape(new_states, shape=(n_x, -1))
            self.set_initial_guess()

        # storing simulation history
        if self.history==None:
            history = {}
            history['x0'] =x0
            history['time'] = [0.0]
            history['u0'] = u0

            self.history = history

        else:
            history = self.history

            history['x0'] = np.hstack([history['x0'], x0])
            history['time'].append(history['time'][-1] + self.t_step)
            history['u0'] = np.hstack([history['u0'], u0])

            self.history = history

        # return predictions
        return x0


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