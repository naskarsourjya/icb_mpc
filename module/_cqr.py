import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ._neuralnetwork import Regressor, MergedModel

class cqr_narx():
    def __init__(self, narx, alpha, n_x, n_u, order, t_step, lbx, ubx, device='auto', set_seed=None, debug=True):
        if set_seed is not None:
            np.random.seed(set_seed)

        # storage
        self.narx = narx
        self.alpha = alpha
        self.n_x = n_x
        self.n_u = n_u
        self.order = order
        self.t_step = t_step
        self.lbx = lbx
        self.ubx = ubx
        self.device = device
        self._cqrstates = None
        self._cqrinputs = None
        self.set_seed = set_seed
        self.history = None
        self.log = None
        self.data = {}

        # setting up default trainer settings
        self.setup_trainer()

        # setting up default plots
        self.setup_plot()

        # flag
        # generating flags
        self.flags = {
            'qr_ready': False,
            'cqr_ready': False,
            'initial_condition_ready': False,
            'debug': debug
        }

        #end


    def _set_device(self, torch_device):

        torch.set_default_device(torch_device)

        return None


    def reshape(self, array, shape):

        # rows and columns
        rows, cols = shape

        # end
        return array.reshape(cols, rows).T


    @property
    def states(self):
        return self._cqrstates


    @states.setter
    def states(self, val):
        assert isinstance(val, np.ndarray), "states must be a numpy.array."

        assert val.shape[1] == self.order, \
            'Number of samples must be equal to the order of the NARX model!'

        assert val.shape[0] == self.n_x, (
            'Expected number of states is: {}, but found {}'.format(self.n_x, val.shape[0]))

        # storage
        self._cqrstates = val


    @property
    def inputs(self):
        return self._cqrinputs


    @inputs.setter
    def inputs(self, val):
        if self.order > 1:
            assert isinstance(val, np.ndarray), "inputs must be a numpy.array."

            assert self.order - 1 == val.shape[1], \
                'Number of samples for inputs should be (order-1) !'

            assert val.shape[0] == self.n_u, (
                'Expected number of inputs is: {}, but found {}'.format(self.n_u, val.shape[0]))

            # storage
            self._cqrinputs = val

        # error
        else:
            raise ValueError("Inputs cannot be set for system with order <= 1.")


    def _pinball_loss(self, y, y_hat, quantile):
        # if y > y_hat:
        #    loss = quantile * (y - y_hat)
        # else:
        #    loss = (1 - quantile) * (y_hat - y)

        # converting to scalar
        # mean_loss = torch.mean(loss)

        diff = y - y_hat
        loss = torch.maximum(quantile * diff, (quantile - 1) * diff)
        mean_loss = loss.mean()

        # end
        return mean_loss


    def setup_trainer(self, hidden_layers=[50, 50, 50], learning_rate=0.1, batch_size=32,
                            validation_split=0.2, scheduler_flag=True, epochs=1000, lr_threshold=1e-8):

        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.scheduler_flag = scheduler_flag
        self.epochs = epochs
        self.lr_threshold = lr_threshold

        return None


    def setup_plot(self, height_px=300, width_px=1800):

        self.height_px = height_px
        self.width_px = width_px

        return None


    def train_individual_qr(self, x_train, y_train):

        # computing calibration error
        error_train = self.surrogate_error(narx_input=x_train, narx_output=y_train)

        # init
        alpha = self.alpha
        n_x = self.n_x
        n_u = self.n_u
        order = self.order
        models = []
        train_history_list = []

        # generating quantiles
        low_quantile = alpha / 2
        high_quantile = 1 - alpha / 2
        quantiles = [high_quantile] + [low_quantile]
        n_q = len(quantiles)

        # scaling data
        scaler = StandardScaler()
        scaler.fit(x_train.T)

        # creating a model for each quantile
        for quantile in quantiles:

            # model init
            cqr_model_n = Regressor(input_size=order * (n_x + n_u),
                                    output_size=n_x,
                                    hidden_layers=self.hidden_layers, scaler=scaler, device=self.device)

            # setting up optimiser for training
            optimizer = torch.optim.AdamW(cqr_model_n.parameters(), lr=self.learning_rate)

            # scheduler setup
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

            # setting training history
            train_history = {'training_loss': [],
                             'validation_loss': [],
                             'learning_rate': [],
                             'epochs': [],
                             'quantile': []}

            # setting computation device
            self._set_device(torch_device=cqr_model_n.torch_device)

            # converting datasets to tensors
            X_torch = torch.tensor(x_train.T, dtype=torch.float32)
            Y_torch = torch.tensor(error_train.T, dtype=torch.float32)

            # Create TensorDataset
            dataset = torch.utils.data.TensorDataset(X_torch, Y_torch)

            # splitting full datset
            train_dataset, validation_dataset = (
                torch.utils.data.random_split(dataset=dataset,
                                              lengths=[1 - self.validation_split, self.validation_split],
                                              generator=torch.Generator(device=cqr_model_n.torch_device).manual_seed(
                                                  self.set_seed)))

            # creating DataLoader with batch_size
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                           generator=torch.Generator(
                                                               device=cqr_model_n.torch_device).manual_seed(
                                                               self.set_seed))
            validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=self.batch_size,
                                                                shuffle=True,
                                                                generator=torch.Generator(
                                                                    device=cqr_model_n.torch_device).manual_seed(
                                                                    self.set_seed))

            # main training loop
            for epoch in tqdm(range(self.epochs), desc=f'Training Cqr q= {quantile}'):

                # cqr training
                train_loss = 0
                for batch_X, batch_Y in train_dataloader:
                    # Forward pass
                    Y_hat = cqr_model_n(batch_X).squeeze()
                    loss = self._pinball_loss(y=batch_Y, y_hat=Y_hat, quantile=quantile)

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
                        Y_hat = cqr_model_n(batch_X).squeeze()
                        val_loss += self._pinball_loss(y=batch_Y, y_hat=Y_hat, quantile=quantile).item()

                # storing data
                train_history['quantile'].append(quantile)
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

            # storage
            models.append(cqr_model_n)
            train_history_list.append(train_history)

        # creating one merged model
        cqr_model = MergedModel(models=models, device=self.device)

        # inserting the mean prediction model
        full_model_list = [self.narx] + models
        full_model = MergedModel(models=full_model_list, device=self.device)

        # store model
        self.cqr_model = cqr_model
        self.scaler = scaler
        self.full_model = full_model
        self.train_history_list = train_history_list
        self.quantiles = quantiles
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.type = 'individual'

        # flag update
        self.flags.update({
            'qr_ready': True,
        })

        return None

    def train_all_qr_depreciate(self, x_train, y_train):

        # init
        models = []
        train_history_list = []

        # generating quantiles
        low_quantile = self.alpha / 2
        high_quantile = 1 - self.alpha / 2
        quantiles = [high_quantile] + [low_quantile]
        n_q = len(quantiles)

        # setting training history
        train_history = {'training_loss': [],
                         'validation_loss': [],
                         'learning_rate': [],
                         'epochs': [],
                         'quantile': []}

        # scaling the input
        scaler = MinMaxScaler()
        scaler.fit(x_train.T)

        # model init
        cqr_model = Regressor(input_size=self.order * (self.n_x + self.n_u),
                              output_size=self.n_x * n_q,
                              hidden_layers=self.hidden_layers, scaler=scaler, device=self.device)

        # setting up optimiser for training
        optimizer = torch.optim.AdamW(cqr_model.parameters(), lr=self.learning_rate)

        # scheduler setup
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        # setting computation device
        self._set_device(torch_device=cqr_model.torch_device)

        # converting datasets to tensors
        X_torch = torch.tensor(x_train.T, dtype=torch.float32)

        # stacking once per quantile
        Y_stacked = np.vstack([y_train for _ in range(n_q)])

        # converting to tensor
        Y_torch = torch.tensor(Y_stacked.T, dtype=torch.float32)

        # Create TensorDataset
        dataset = torch.utils.data.TensorDataset(X_torch, Y_torch)

        # splitting full datset
        train_dataset, validation_dataset = (
            torch.utils.data.random_split(dataset=dataset, lengths=[1 - self.validation_split, self.validation_split],
                                          generator=torch.Generator(device=cqr_model.torch_device).manual_seed(
                                              self.set_seed)))

        # creating DataLoader with batch_size
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                       generator=torch.Generator(
                                                           device=cqr_model.torch_device).manual_seed(
                                                           self.set_seed))
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True,
                                                            generator=torch.Generator(
                                                                device=cqr_model.torch_device).manual_seed(
                                                                self.set_seed))

        # main training loop
        for epoch in tqdm(range(self.epochs), desc=f'Training All CQR'):

            # cqr training
            train_loss = 0
            for batch_X, batch_Y in train_dataloader:
                # Forward pass
                Y_hat = cqr_model(batch_X).squeeze()
                loss = 0
                for quantile in quantiles:
                    loss += self._pinball_loss(y=batch_Y, y_hat=Y_hat, quantile=quantile)

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
                    Y_hat = cqr_model(batch_X).squeeze()
                    for quantile in quantiles:
                        val_loss += self._pinball_loss(y=batch_Y, y_hat=Y_hat, quantile=quantile).item()

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

        # storage
        train_history_list.append(train_history)

        # inserting the mean prediction model
        full_model_list = [self.narx['model']] + [cqr_model]
        full_model = MergedModel(models=full_model_list, device=self.device)

        # store model
        self.cqr_model = cqr_model
        self.scaler = scaler
        self.full_model = full_model
        self.train_history_list = train_history_list
        self.quantiles = quantiles
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.type = 'all'

        # flag update
        self.flags.update({
            'qr_ready': True,
        })

        # end
        return None


    def surrogate_error(self, narx_input, narx_output):

        # setting default device
        self._set_device(torch_device=self.narx.torch_device)

        # preprosecssing
        X_torch = torch.tensor(narx_input.T, dtype=torch.float32)

        # making full model prediction
        with torch.no_grad():
            narx_pred = self.narx(X_torch).cpu().numpy().T

        narx_error = narx_output-narx_pred

        return narx_error

    def conform_qr(self, x_calib, y_calib):
        assert self.flags['qr_ready'] == True, \
            'CQR not found! Train or load CQR model!'

        # computing calibration error
        error_calib = self.surrogate_error(narx_input=x_calib, narx_output=y_calib)

        # storage in convenient varaibles
        n_x = self.n_x
        quantiles = self.quantiles
        low_quantile = self.low_quantile
        high_quantile = self.high_quantile
        alpha = self.alpha
        n_samples = error_calib.shape[1]

        # scaling calibration data
        #x_calib_sc = self.scaler.transform(x_calib.T)

        # making quantile prediction
        Xi_troch = torch.tensor(x_calib.T, dtype=torch.float32)
        with torch.no_grad():
            qr_all = self.cqr_model(Xi_troch).cpu().numpy().T

        index_high = quantiles.index(high_quantile)
        index_low = quantiles.index(low_quantile)

        # storing the values
        q_lo = qr_all[n_x * index_low: n_x + n_x * index_low, :]
        q_hi = qr_all[n_x * index_high: n_x + n_x * index_high, :]

        for j in range(n_x):

            # conformalising one state at a time
            q_lo_xn = q_lo[j,:]
            q_hi_xn = q_hi[j, :]
            Yi_xn = error_calib[j,:]

            # Generating conformity scores
            Ei_xn = np.max(np.vstack([q_lo_xn - Yi_xn, Yi_xn - q_hi_xn]), axis = 0)

            # calculating the appropriate quantile
            error_quantile = (1 - alpha) * (1 + 1/n_samples)

            # Compute the quantile
            Q_xn = np.quantile(Ei_xn, q=error_quantile)

            # storage
            if j == 0:
                    Q1_alpha = Q_xn
            else:
                Q1_alpha = np.vstack([Q1_alpha, Q_xn])

        # storage
        self.Q1_alpha = torch.tensor(Q1_alpha, dtype=torch.float32)
        self.data['cqr_calibration_inputs'] = x_calib
        self.data['cqr_calibration_outputs'] = y_calib
        self.data['cqr_calibration_errors'] = error_calib

        # update flag
        self.flags.update({
            'cqr_ready': True
        })

        # end
        return None


    def set_initial_guess(self):

        assert self.flags['qr_ready'] == True, \
            'CQR not trained! Train CQR model!'

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
            'cqr_initial_condition_ready': True
        })

        # end
        return None


    def make_step(self, u0):
        assert self.flags['qr_ready'], "Qunatile regressor not ready."
        assert self.flags['cqr_ready'], "Qunatile regressor not conformalised."
        assert self.flags['cqr_initial_condition_ready'], "CQR not initialised"
        assert u0.shape[0] == self.n_u, \
            f"u0 should have have {self.n_u} rows but instead found {u0.shape[0]}!"
        assert u0.shape[1] == 1, \
            f"u0 should have have 1 columns but instead found {u0.shape[1]}!"

        # init
        initial_cond = self.initial_cond
        n_x = self.n_x
        n_u = self.n_u
        order = self.order


        # segregating states and inputs
        states = initial_cond[0:n_x*order, :]
        inputs = initial_cond[n_x*order:, :]

        # stacking all data
        X = np.vstack([states, u0, inputs])

        # scaling
        #X_scaled = self.scaler.transform(X.T)

        # setting default device
        self._set_device(torch_device=self.full_model.torch_device)

        # narx_input = self.input_preprocessing(states=order_states, inputs=order_inputs)
        X_torch = torch.tensor(X.T, dtype=torch.float32)

        # making full model prediction
        with torch.no_grad():
            y_pred = self.full_model(X_torch).T

        # doing postprocessing containing the conformalisation step
        x0, x0_cqr_high, x0_cqr_low = self._post_processing(y=y_pred)

        # pushing oldest state out of system and inserting the current state
        new_states = np.vstack([states[n_x:(order)*n_x, :], x0])

        if order>1:

            # pushing oldest input out of system and inserting the current input
            new_inputs = np.vstack([inputs[n_u:(order-1)*n_u, :], u0])

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
            history['x0_cqr'] =x0
            history['x0_cqr_high'] = x0_cqr_high
            history['x0_cqr_low'] = x0_cqr_low
            history['time'] = [0.0]
            history['u0'] = u0

            self.history = history

        else:
            history = self.history

            history['x0_cqr'] = np.hstack([history['x0_cqr'], x0])
            history['x0_cqr_high'] = np.hstack([history['x0_cqr_high'], x0_cqr_high])
            history['x0_cqr_low'] = np.hstack([history['x0_cqr_low'], x0_cqr_low])
            history['time'].append(history['time'][-1] + self.t_step)
            history['u0'] = np.hstack([history['u0'], u0])

            self.history = history

        # logged val
        if self.flags['debug']:
            # storing simulation history
            if self.log == None:
                log = {}
                log['X'] = X
                log['time'] = [0.0]
                self.log = log

            else:
                log = self.log
                log['X'] = np.hstack([log['X'], X])
                log['time'].append(log['time'][-1] + self.t_step)
                self.log = log

        # return predictions
        return x0, x0_cqr_high, x0_cqr_low


    def export_log(self, file_name = 'CQR_NARX Model Log.csv'):

        assert self.flags['debug'], "Data not logged! Enable debug=True to access log."

        # generating names for the dataframe
        state_names = []
        input_names = []
        for o in range(self.order):
            for n_xn in range(self.n_x):
                state_name = f'state_{n_xn+1}_lag_{self.order-1-o}'
                state_names.append(state_name)

            for n_un in range(self.n_u):
                input_name = f'input_{n_un+1}_lag_{self.order-1-o}'
                input_names.append(input_name)

        # dataset
        col_names = ['time'] + state_names + input_names
        data = np.hstack([np.vstack(self.log['time']),
                          self.log['X'].T])

        # Converting to dataframe
        df = pd.DataFrame(data, columns=col_names)

        # saving dataframe
        df.to_csv(file_name, index=False)

        # end
        return None


    def _post_processing_old(self, y):

        y = y.cpu().numpy()

        # init
        n_x = self.n_x
        Q1_alpha = self.Q1_alpha
        n_samples = y.shape[1]

        # stacking states
        states = y[0:n_x,:]
        stacked_states = np.vstack([states] * 3)

        # stacking errors
        errors = y[n_x:, :]
        stacked_errors = np.vstack([np.zeros((n_x, n_samples)), errors])

        # stacking conformalisation
        conform = np.hstack([Q1_alpha] * n_samples)
        stacked_confrom = np.vstack([np.zeros((n_x, n_samples)), conform, -conform])

        # final prediction
        pred = stacked_states + stacked_errors + stacked_confrom

        # extracting the quantiles
        x0 = pred[0:n_x, :]
        x0_cqr_high = pred[n_x:2 * n_x, :]
        x0_cqr_low = pred[2 * n_x:, :]

        # end
        return x0, x0_cqr_high, x0_cqr_low


    def _post_processing(self, y):

        # init
        n_x = self.n_x
        Q1_alpha = self.Q1_alpha
        n_samples = y.shape[1]

        # stacking states
        states = y[0:n_x,:]
        stacked_states = torch.vstack([states] * 3)

        # stacking errors
        errors = y[n_x:, :]
        stacked_errors = torch.vstack([torch.zeros((n_x, n_samples)), errors])

        # stacking conformalisation
        conform = torch.hstack([Q1_alpha] * n_samples)
        stacked_confrom = torch.vstack([torch.zeros((n_x, n_samples)), conform, -conform])

        # final prediction
        pred = stacked_states + stacked_errors + stacked_confrom

        # extracting the quantiles
        x0 = pred[0:n_x, :]
        x0_cqr_high = pred[n_x:2 * n_x, :]
        x0_cqr_low = pred[2 * n_x:, :]

        # end
        return x0.cpu().numpy(), x0_cqr_high.cpu().numpy(), x0_cqr_low.cpu().numpy()


    def plot_qr_training_history(self):
        assert self.flags['qr_ready'] == True, 'CQR not found! Generate or load CQR model!'


        if self.type == 'individual':

            quantiles = self.quantiles
            n_q = len(quantiles)

            # plot init
            fig = make_subplots(
                rows=n_q, cols=2,
                shared_xaxes=True,
                subplot_titles=['Learning rate', 'Loss'],
                specs=[[{}, {"secondary_y": True}] for _ in quantiles],
                # Enable secondary y-axis in column 2
                row_heights=[0.5] * n_q,  # Adjust row heights for better layout
                column_widths=[1] * 2
            )

            # updating layout
            fig.update_layout(title_text='Individual CQR Training History',
                              height=self.height_px * n_q, width=self.width_px)

            # making plots
            for i, quantile in enumerate(quantiles):
                # Extracting history
                training_history = self.train_history_list[i]

                # Plot 1: Learning Rate (left column)
                fig.add_trace(go.Scatter(x=training_history['epochs'], y=training_history['learning_rate'],
                                         mode='lines', line=dict(color='red'),
                                         name='learning rate',
                                         showlegend=False),
                              row=i + 1, col=1)
                fig.update_yaxes(type='log', title_text=f'CQR (q={quantile})\nLearning Rate', row=i + 1, col=1)
                fig.update_xaxes(title_text='epochs', row=i + 1, col=1)

                # Plot 2: Training Loss (primary y-axis in right column)
                fig.add_trace(go.Scatter(x=training_history['epochs'], y=training_history['training_loss'],
                                         mode='lines', line=dict(color='green'),
                                         name='training loss',
                                         showlegend=True if i == 0 else False),
                              row=i + 1, col=2)
                fig.update_yaxes(type='log', title_text=f'CQR (q={quantile})\nTraining Loss', row=i + 1, col=2)

                # Validation Loss (secondary y-axis in right column)
                fig.add_trace(go.Scatter(x=training_history['epochs'], y=training_history['validation_loss'],
                                         mode='lines', line=dict(color='blue'),
                                         name = 'validation loss',
                                         showlegend=True if i == 0 else False),
                              row=i + 1, col=2, secondary_y=True)
                fig.update_yaxes(title_text=f'CQR (q={quantile})\nValidation Loss', type='log',
                                 row=i + 1, col=2, secondary_y=True)
                fig.update_xaxes(title_text='epochs', row=i + 1, col=2)

            fig.show()

        elif self.type == 'all':
            # Create subplots with secondary_y set in row 2
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                subplot_titles=['Loss', 'Learning Rate'],
                specs=[[{"secondary_y": True}], [{"secondary_y": False}]]  # Enable secondary y-axis only in row 2
            )

            fig.update_layout(title_text='All CQR Training History', height=self.height_px, width=self.width_px)

            # Extracting history
            training_history = self.train_history_list[0]

            # Plot 1: Training Loss (primary y-axis in row 1)
            fig.add_trace(go.Scatter(x=training_history['epochs'], y=training_history['training_loss'],
                                     mode='lines', line=dict(color='green'),
                                     name=f'training loss',
                                     showlegend=True),
                          row=1, col=1)
            fig.update_yaxes(type='log', title_text='Training Loss', row=1, col=1)

            # Validation Loss (secondary y-axis in row 1)
            fig.add_trace(go.Scatter(x=training_history['epochs'], y=training_history['validation_loss'],
                                     mode='lines', line=dict(color='red'),
                                     name=f'validation loss',
                                     showlegend=True),
                          row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text='Validation Loss', type='log', row=1, col=1, secondary_y=True)
            fig.update_xaxes(title_text='epochs', row=1, col=1)

            # Plot 2: Learning Rate (row 2)
            fig.add_trace(go.Scatter(x=training_history['epochs'], y=training_history['learning_rate'],
                                     mode='lines', line=dict(color='blue'),
                                     showlegend=False),
                          row=2, col=1)
            fig.update_yaxes(type='log', title_text='Learning Rate', row=2, col=1)
            fig.update_xaxes(title_text='epochs', row=2, col=1)

            fig.show()

        return None

    def plot_qr_error(self, t_test):

        assert self.flags['qr_ready'], "Qunatile regressor not ready."

        x_test = self.data['cqr_calibration_inputs']
        y_test = self.data['cqr_calibration_errors']

        # init
        n_x = self.n_x
        quantiles = self.quantiles
        n_q = len(quantiles)
        low_quantile = self.low_quantile
        high_quantile = self.high_quantile
        n_a = 1

        # scaling
        #X_scaled = self.scaler.transform(x_test.T)

        # setting default device
        self._set_device(torch_device=self.cqr_model.torch_device)

        # narx_input = self.input_preprocessing(states=order_states, inputs=order_inputs)
        X_narx = torch.tensor(x_test.T, dtype=torch.float32)

        # making prediction
        with torch.no_grad():
            Y_pred = self.cqr_model(X_narx).cpu().numpy().T

        # setting up plots
        #fig, ax = plt.subplots(n_x, figsize=(24, 6 * n_x))
        #fig.suptitle('QR Error plots')

        fig = make_subplots(rows=n_x, cols=1, shared_xaxes=True)
        fig.update_layout(height=self.height_px * n_x, width=self.width_px, title_text="QR Error Plots",
                          showlegend=True)

        # sorting with timestamps
        x = t_test.reshape(-1, )
        sorted_indices = np.argsort(x)  # Get indices that would sort x
        x_sorted = x[sorted_indices]

        # plot for each state
        for i in range(n_x):
            # plot the real mean
            #ax[i].plot(x_sorted, y_calib[i, :][sorted_indices], label=f'real mean')
            fig.add_trace(go.Scatter(x=x_sorted, y=y_test[i, :][sorted_indices],
                                     mode='lines', line=dict(color='red'),
                                     name='real mean',
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)
            fig.update_yaxes(title_text=f' State {i+1}', row=i + 1, col=1)
            fig.update_xaxes(title_text='Times Stamp [s]', row=i + 1, col=1)

            for j in range(n_q):

                index = i + n_x * j

                # plotting cqr high side
                if j < n_a:
                    #ax[i].plot(x_sorted, Y_pred[index, :][sorted_indices], label=f'quantile={high_quantile}')
                    fig.add_trace(go.Scatter(x=x_sorted, y=Y_pred[index, :][sorted_indices],
                                             mode='lines', line=dict(color='green'),
                                             name=f'quantile={high_quantile}',
                                             showlegend=True if i == 0 else False),
                                  row=i + 1, col=1)
                    #fig.update_yaxes(title_text=f' State {i + 1}', row=i + 1, col=1)
                    #fig.update_xaxes(title_text='Times Stamp [s]', row=i + 1, col=1)

                # plotting cqr low side
                elif j >= n_a:
                    #ax[i].plot(x_sorted, Y_pred[index, :][sorted_indices], label=f'quantile={low_quantile}')
                    fig.add_trace(go.Scatter(x=x_sorted, y=Y_pred[index, :][sorted_indices],
                                             mode='lines', line=dict(color='blue'),
                                             name=f'quantile={low_quantile}',
                                             showlegend=True if i == 0 else False),
                                  row=i + 1, col=1)
                    #fig.update_yaxes(title_text=f' State {i + 1}', row=i + 1, col=1)
                    #fig.update_xaxes(title_text='Times Stamp [s]', row=i + 1, col=1)

        # show plot
        fig.show()

        # end
        return None

    # Function to plot CQR error using Plotly
    def plot_cqr_error(self, x_test, y_test, t_test):
        assert self.flags['qr_ready'], "Quantile regressor not ready."
        assert self.flags['cqr_ready'], "Quantile regressor not conformalised."

        # Init
        order = self.order
        n_x = self.n_x
        n_u = self.n_u
        low_quantile = self.low_quantile
        high_quantile = self.high_quantile
        n_samples = x_test.shape[1]
        alpha = self.alpha

        # Calculating model intervals
        for i in tqdm(range(n_samples), desc='Calculating surrogate model state intervals'):
            states_history = x_test[0:n_x * order, i]
            inputs_n = x_test[n_x * order:, i]
            u0 = inputs_n[0:n_u]
            inputs_history = inputs_n[n_u:]

            if order > 1:
                self.states=self.reshape(states_history, shape=(n_x, -1))
                self.inputs=self.reshape(inputs_history, shape=(n_u, -1))
                self.set_initial_guess()
                x0, x0_cqr_high, x0_cqr_low = self.make_step(u0=self.reshape(u0, shape=(n_u, 1)))
            else:
                self.states=self.reshape(states_history, shape=(n_x, -1))
                self.set_initial_guess()
                x0, x0_cqr_high, x0_cqr_low = self.make_step(u0=self.reshape(u0, shape=(n_u, 1)))

            if i == 0:
                Y_predicted_mean = x0
                Y_predicted_high = x0_cqr_high
                Y_predicted_low = x0_cqr_low
            else:
                Y_predicted_mean = np.hstack([Y_predicted_mean, x0])
                Y_predicted_high = np.hstack([Y_predicted_high, x0_cqr_high])
                Y_predicted_low = np.hstack([Y_predicted_low, x0_cqr_low])

        # Sorting according to timestamps
        x = t_test.reshape(-1, )
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]

        # Create subplots
        fig = make_subplots(rows=n_x, cols=1, shared_xaxes=True)
        fig.update_layout(height=self.height_px * n_x, width=self.width_px, title_text="CQR State Plots",
                          showlegend=True)

        # Loop through each state
        for i in range(n_x):

            # Add shaded region for bounds
            fig.add_trace(go.Scatter(
                x=[min(x_sorted), max(x_sorted), max(x_sorted), min(x_sorted)],
                y=[self.lbx[i], self.lbx[i], self.ubx[i], self.ubx[i]],
                fill="toself",
                fillcolor="rgba(200, 200, 200, 0.3)",  # Grey shaded region
                line=dict(color="rgba(255,255,255,0)"),  # No border
                name="Bounds", showlegend=True if i == 0 else False),
                row=i + 1, col=1)

            # Shaded confidence interval (show legend for the first plot of each row)
            fig.add_trace(go.Scatter(x=np.concatenate((x_sorted, x_sorted[::-1])),
                                     y=np.concatenate((Y_predicted_high[i, sorted_indices],
                                                       Y_predicted_low[i, sorted_indices][::-1])),
                                     fill='toself', fillcolor='rgba(128, 128, 128, 0.5)',
                                     line=dict(color='rgba(255,255,255,0)'),
                                     name=f'Confidence {1 - alpha}',
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            # Add lines for upper and lower bounds
            fig.add_trace(go.Scatter(x=x_sorted, y=[self.ubx[i]] * len(x_sorted), mode='lines',
                                     line=dict(color='red', dash='dash'), name='Upper Bound',
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)
            fig.add_trace(go.Scatter(x=x_sorted, y=[self.lbx[i]] * len(x_sorted), mode='lines',
                                     line=dict(color='green', dash='dash'), name='Lower Bound',
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            # Predicted mean line (show legend for the first plot of each row)
            fig.add_trace(go.Scatter(x=x_sorted, y=Y_predicted_mean[i, sorted_indices],
                                     mode='lines', name=f'Predicted Mean',
                                     line=dict(color='blue', dash='longdashdot'),
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            # Real mean line (show legend for the first plot of each row)
            fig.add_trace(go.Scatter(x=x_sorted, y=y_test[i, sorted_indices],
                                     mode='lines', name=f'Real Mean',
                                     line=dict(color='orange'),
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            # CQR High quantile (show legend for the first plot of each row)
            fig.add_trace(go.Scatter(x=x_sorted, y=Y_predicted_high[i, sorted_indices],
                                     mode='markers', name=f'High Quantile={high_quantile}',
                                     marker=dict(color='green', size=6),
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            # CQR Low quantile (show legend for the first plot of each row)
            fig.add_trace(go.Scatter(x=x_sorted, y=Y_predicted_low[i, sorted_indices],
                                     mode='markers', name=f'Low Quantile={low_quantile}',
                                     marker=dict(color='purple', size=6),
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            fig.update_yaxes(title_text=f' State {i + 1}', row=i + 1, col=1)
            fig.update_xaxes(title_text='Times Stamp [s]', row=i + 1, col=1)

        # Show plot
        fig.show()
        return None


    def make_branch(self, u0_traj, confidence_cutoff=0.5):
        assert self.flags['qr_ready'], "Qunatile regressor not ready."
        assert self.flags['cqr_ready'], "Qunatile regressor not conformalised."
        assert self.flags['cqr_initial_condition_ready'], "CQR not initialised"
        assert u0_traj.shape[0] == self.n_u, \
            f"u0 should have have {self.n_u} rows but instead found {u0_traj.shape[0]}!"

        # storage for later retreival
        n_x = self.n_x
        n_u = self.n_u
        order = self.order
        prev_ic = self.initial_cond
        current_ic = self.initial_cond
        states_branch = [self.initial_cond[0:self.n_x]]
        alpha_branch = [1-self.alpha]
        time_branch = [0.0]
        steps = u0_traj.shape[1]

        # generating the branches
        for i in range(steps):

            # init
            u0 = self.reshape(u0_traj[:,i], shape=(n_u, 1))
            n_samples = current_ic.shape[1]

            # segregating states and inputs
            states = current_ic[0:n_x * order, :]
            inputs = current_ic[n_x * order:, :]
            u0_stacked = np.hstack([u0] * n_samples)

            # stacking all data
            X = np.vstack([states, u0_stacked, inputs])

            # scaling
            #X_scaled = self.scaler.transform(X.T)

            # setting default device
            self._set_device(torch_device=self.full_model.torch_device)

            # narx_input = self.input_preprocessing(states=order_states, inputs=order_inputs)
            X_torch = torch.tensor(X.T, dtype=torch.float32)

            # making full model prediction
            with torch.no_grad():
                y_pred = self.full_model(X_torch).T

            # doing postprocessing containing the conformalisation step
            x0, x0_cqr_high, x0_cqr_low = self._post_processing(y=y_pred)

            # generate new current_ic for next branch
            # mean side
            next_mean_ic = np.vstack([x0, states[0:n_x*(order-1), :], u0_stacked, inputs[0:n_u*(order-2), :]])

            # high side
            next_high_ic = np.vstack([x0_cqr_high, states[0:n_x*(order-1), :], u0_stacked, inputs[0:n_u*(order-2), :]])

            #low side
            next_low_ic = np.vstack([x0_cqr_low, states[0:n_x*(order-1), :], u0_stacked, inputs[0:n_u*(order-2), :]])

            # preparing ic's for the next iteration
            current_ic = np.hstack([next_mean_ic, next_high_ic, next_low_ic])

            # stores the branched states
            states_branch.append(np.hstack([x0, x0_cqr_high, x0_cqr_low]))
            alpha_branch.append(alpha_branch[-1]*(1-self.alpha))
            time_branch.append(time_branch[-1] + self.t_step)

            # force cutoff is confidence is low
            if alpha_branch[-1] < confidence_cutoff:
                break

        # reverting back to previous ic
        self.initial_cond = prev_ic
        self.confidence_cutoff = confidence_cutoff

        # storage
        self.branches = {'states': states_branch,
                         'alphas': alpha_branch,
                         'time_stamps': time_branch,
                         'u0_traj': u0_traj}

        # end
        return self.branches


    def plot_branch(self, t0=0.0, show_plot=True):

        n_x = self.n_x
        n_u = self.n_u
        time_stamps = [num + t0 for num in self.branches['time_stamps']]
        states = self.branches['states']
        alphas = self.branches['alphas']
        u0_traj = self.branches['u0_traj']

        # Create subplots

        fig = make_subplots(rows=n_x+n_u, cols=1, shared_xaxes=True)
        fig.update_layout(height=self.height_px * (n_x+n_u), width=self.width_px, title_text="CQR State Branch Plots",
                              showlegend=True)

        # Loop through each state
        for i in range(n_x):
            # init
            mean_prediction = []

            # Predicted mean line (show legend for the first plot of each row)
            for j, t in enumerate(time_stamps):

                if j<len(time_stamps)-1:
                    # Add shaded region for confidence
                    fig.add_trace(go.Scatter(
                        x=[time_stamps[j], time_stamps[j+1], time_stamps[j+1], time_stamps[j]],
                        y=[min(states[j][i,:]), min(states[j+1][i,:]), max(states[j+1][i,:]), max(states[j][i,:])],
                        fill="toself",
                        fillcolor=f"rgba(255, 255, 0, {alphas[j]})",  # Grey shaded region
                        line=dict(color="rgba(255,255,255,0)"),  # No border
                        name=f"Confidence={alphas[j]}", showlegend=False),

                        row=i + 1, col=1)


                fig.add_trace(go.Scatter(x=[t]*states[j][i,:].shape[0], y=states[j][i,:],
                                         mode='markers', name=f'Branches',
                                         marker=dict(color='pink', size=2),
                                         showlegend=True if i==0 and j==0 else False),
                              row=i + 1, col=1)

                # extracting mean prediction
                mean_prediction.append(states[j][i, 0])

            # making the mean prediction
            fig.add_trace(go.Scatter(x=time_stamps, y=mean_prediction,
                                     mode='lines',
                                     line=dict(color='red', dash='dash'), name='Nominal Projection',
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            fig.update_yaxes(title_text=f' State {i + 1}', row=i + 1, col=1)
            fig.update_xaxes(title_text='Times [s]', row=i + 1, col=1)

        for j in range(n_u):
            # making the mean prediction
            fig.add_trace(go.Scatter(x=time_stamps, y=u0_traj[j,:],
                                     mode='lines',
                                     line=dict(color='red', dash='dash'), name='Input Trajectory',
                                     showlegend=False),
                          row=j+i+2, col=1)

            fig.update_yaxes(title_text=f' Inputs {j + 1}', row=j+i+2, col=1)
            fig.update_xaxes(title_text='Times [s]', row=j+i+2, col=1)

        # Show plot
        if show_plot:
            fig.show()
            return None

        # returns plot for further modifiction
        else:
            return fig






















