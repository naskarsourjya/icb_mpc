import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

from ._neuralnetwork import Regressor, MergedModel

class cqr_narx():
    def __init__(self, narx, alpha, n_x, n_u, order, t_step, lbx, ubx, lbu, ubu, device='auto', set_seed=0, debug=True, dtype=torch.float64):
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
        self.lbu = lbu
        self.ubu = ubu
        self.device = device
        self._cqrstates = None
        self._cqrinputs = None
        self.set_seed = set_seed
        self.history = None
        self.log = None
        self.data = {}
        self.dtype = dtype

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


    def set_config(self, rnd_samples=7, confidence_cutoff=0.5):
        assert isinstance(rnd_samples, int) and rnd_samples>=0, "Only positive integers permissible."
        assert confidence_cutoff>0 and confidence_cutoff<=1, " Only positive fraction less than equal to 1 permissible."
        self.rnd_samples = rnd_samples
        self.confidence_cutoff = confidence_cutoff
        return None


    def _set_device(self, torch_device):

        # set default device
        torch.set_default_device(torch_device)

        # set default dtype
        torch.set_default_dtype(self.dtype)

        return None


    @property
    def states(self):
        return self._cqrstates


    @states.setter
    def states(self, val):
        assert isinstance(val, np.ndarray), "states must be a numpy.array."

        assert val.shape[0] == self.order, \
            'Number of samples must be equal to the order of the NARX model!'

        assert val.shape[1] == self.n_x, (
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

            assert self.order - 1 == val.shape[0], \
                'Number of samples for inputs should be (order-1) !'

            assert val.shape[1] == self.n_u, (
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
                            validation_split=0.2, scheduler_flag=True, epochs=1000, lr_threshold=1e-8,
                      train_threshold= None):

        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.scheduler_flag = scheduler_flag
        self.epochs = epochs
        self.lr_threshold = lr_threshold
        self.train_threshold = train_threshold

        return None


    def setup_plot(self, height_px=9, width_px=16):

        self.height_px = height_px
        self.width_px = width_px

        return None


    def train(self, x_train, y_train):

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
        input_scaler = StandardScaler()
        x_train_scaled = input_scaler.fit_transform(x_train)

        output_scaler = StandardScaler()
        error_train_scaled = output_scaler.fit_transform(error_train)

        # creating a model for each quantile
        for quantile in quantiles:

            # model init
            cqr_model_n = Regressor(input_size=order * (n_x + n_u),
                                    output_size=n_x,
                                    hidden_layers=self.hidden_layers, input_scaler=input_scaler,
                                    output_scaler=output_scaler, device=self.device)

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
            X_torch = torch.tensor(x_train_scaled, dtype=self.dtype)
            Y_torch = torch.tensor(error_train_scaled, dtype=self.dtype)

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
                    train_loss += loss.item()/len(train_dataset)

                # narx validation
                val_loss = 0
                for batch_X, batch_Y in validation_dataloader:
                    with torch.no_grad():
                        Y_hat = cqr_model_n(batch_X).squeeze()
                        val_loss += self._pinball_loss(y=batch_Y, y_hat=Y_hat, quantile=quantile).item()/len(validation_dataset)

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

                # break if training threshold is reached
                if self.train_threshold is not None and train_loss < self.train_threshold:
                    break

            # storage
            models.append(cqr_model_n)
            train_history_list.append(train_history)

        # creating one merged model
        cqr_model = MergedModel(models=models, device=self.device)

        # inserting the mean prediction model
        full_model_list = [self.narx.model] + models
        full_model = MergedModel(models=full_model_list, device=self.device)

        # store model
        self.cqr_model = cqr_model
        self.cqr_high_model = models[0]
        self.cqr_low_model = models[1]
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
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


    def surrogate_error(self, narx_input, narx_output):

        # setting default device
        self._set_device(torch_device=self.narx.model.torch_device)

        # preprosecssing
        X_torch = torch.tensor(narx_input.to_numpy(), dtype=self.dtype)

        # making full model prediction
        with torch.no_grad():
            narx_pred = self.narx.model.evaluate(X_torch).cpu().numpy()

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
        n_samples = error_calib.shape[0]

        # scaling calibration data

        # making quantile prediction
        Xi_troch = torch.tensor(x_calib.to_numpy(), dtype=self.dtype)
        with torch.no_grad():
            qr_all = self.cqr_model(Xi_troch).cpu().numpy()

        index_high = quantiles.index(high_quantile)
        index_low = quantiles.index(low_quantile)

        # storing the values
        q_lo = qr_all[:, n_x * index_low: n_x + n_x * index_low]
        q_hi = qr_all[:, n_x * index_high: n_x + n_x * index_high]

        for j in range(n_x):

            # conformalising one state at a time
            q_lo_xn = q_lo[:, j]
            q_hi_xn = q_hi[:, j]
            Yi_xn = error_calib.to_numpy()[:, j]

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
                Q1_alpha = np.hstack([Q1_alpha, Q_xn])

        # storage
        self.Q1_alpha = torch.tensor(Q1_alpha, dtype=self.dtype)
        self.data['cqr_calibration_inputs'] = x_calib
        self.data['cqr_calibration_outputs'] = y_calib
        self.data['cqr_calibration_errors'] = error_calib

        # update flag
        self.flags.update({
            'cqr_ready': True
        })

        # end
        return None


    def _set_initial_guess(self):

        assert self.flags['qr_ready'] == True, \
            'CQR not trained! Train CQR model!'

        assert self.flags['cqr_ready'] == True, \
            'QR not confromalised! Conformalise QR model.'

        # init
        states = self.states
        inputs = self.inputs

        # reshaping
        init_states = states.reshape((1, -1))

        # storage
        self.init_states = init_states
        self.init_inputs = None

        if self.order>1:
            init_inputs = inputs.reshape((1, -1))
            self.init_inputs = init_inputs

        # flag update
        self.flags.update({
            'cqr_initial_condition_ready': True
        })

        # end
        return None


    def set_initial_guess(self):

        # setting up the states and the inputs
        self._set_initial_guess()

        # init time
        self.t0 = 0.0

        # init storage
        history = {}
        history['x0_cqr'] = self.states[0, :].reshape((1, -1))
        history['x0_cqr_high'] = np.full((1, self.n_x), np.nan)
        history['x0_cqr_low'] = np.full((1, self.n_x), np.nan)
        history['time'] = np.array([[self.t0]])
        history['u0'] = np.full((1, self.n_u), np.nan)
        self.history = history

        # end
        return None



    def make_step(self, u0):
        assert self.flags['qr_ready'], "Qunatile regressor not ready."
        assert self.flags['cqr_ready'], "Qunatile regressor not conformalised."
        assert self.flags['cqr_initial_condition_ready'], "CQR not initialised"
        assert u0.shape[0] == 1, \
            f"u0 should have have 1 row but instead found {u0.shape[0]}!"
        assert u0.shape[1] == self.n_u, \
            f"u0 should have have {self.n_u} columns but instead found {u0.shape[1]}!"

        # init
        n_x = self.n_x
        n_u = self.n_u
        order = self.order

        # stacking all data
        if self.order>1:
            X = np.hstack([self.init_states, u0, self.init_inputs])
        else:
            X = np.hstack([self.init_states, u0])

        # setting default device
        self._set_device(torch_device=self.full_model.torch_device)

        # narx_input = self.input_preprocessing(states=order_states, inputs=order_inputs)
        X_torch = torch.tensor(X, dtype=self.dtype)

        # making full model prediction
        with torch.no_grad():
            y_pred = self.full_model(X_torch)

        # doing postprocessing containing the conformalisation step
        x0, x0_cqr_high, x0_cqr_low = self._post_processing(y=y_pred)

        # pushing oldest state out of system and inserting the current state
        new_states = np.vstack([x0, self.states[:order-1, :]])

        if order>1:

            # pushing oldest input out of system and inserting the current input
            new_inputs = np.vstack([u0, self. inputs[:order-2, :]])

            # setting new initial guess by removing the last timestamp data
            self.states = new_states
            self.inputs = new_inputs
            self._set_initial_guess()

        else:
            self.states = new_states
            self._set_initial_guess()

        # stepping up time
        self.t0 = self.t0 + self.t_step

        # storing simulation history
        history = self.history
        history['x0_cqr'] = np.vstack([history['x0_cqr'], x0])
        history['x0_cqr_high'] = np.vstack([history['x0_cqr_high'], x0_cqr_high])
        history['x0_cqr_low'] = np.vstack([history['x0_cqr_low'], x0_cqr_low])
        history['time'] = np.vstack([history['time'], self.t0])
        history['u0'] = np.vstack([history['u0'], u0])
        self.history = history

        # logged val
        if self.flags['debug']:
            # storing simulation history
            if self.log == None:
                log = {}
                log['X'] = X
                log['time'] = np.array([[self.t0]])
                self.log = log

            else:
                log = self.log
                log['X'] = np.vstack([log['X'], X])
                log['time'] = np.vstack([log['time'], self.t0])
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
                state_name = f'state_{n_xn+1}_lag_{o+1}'
                state_names.append(state_name)

            for n_un in range(self.n_u):
                input_name = f'input_{n_un+1}_lag_{o}'
                input_names.append(input_name)

        # dataset
        col_names = ['time'] + state_names + input_names
        data = np.hstack([np.vstack(self.log['time']),
                          self.log['X']])

        # Converting to dataframe
        df = pd.DataFrame(data, columns=col_names)

        # saving dataframe
        df.to_csv(file_name, index=False)

        # end
        return None


    def _post_processing(self, y):

        # init
        n_x = self.n_x
        Q1_alpha = self.Q1_alpha
        n_samples = y.shape[0]

        # stacking states
        states = y[:, 0:n_x]
        stacked_states = torch.hstack([states] * 3)

        # stacking errors
        errors = y[:, n_x:]
        stacked_errors = torch.hstack([torch.zeros((n_samples, n_x)), errors])

        # stacking conformalisation
        conform = torch.vstack([Q1_alpha] * n_samples)
        stacked_confrom = torch.hstack([torch.zeros((n_samples, n_x)), conform, -conform])

        # final prediction
        pred = stacked_states + stacked_errors + stacked_confrom

        # extracting the quantiles
        x0 = pred[:, 0:n_x]
        x0_cqr_high = pred[:, n_x:2 * n_x]
        x0_cqr_low = pred[:, 2 * n_x:]

        # end
        return x0.cpu().numpy(), x0_cqr_high.cpu().numpy(), x0_cqr_low.cpu().numpy()


    def plot_qr_training_history(self):
        assert self.flags['qr_ready'] is True, 'CQR not found! Generate or load CQR model!'

        if self.type == 'individual':
            quantiles = self.quantiles
            n_q = len(quantiles)

            fig, axs = plt.subplots(n_q, 2, figsize=(self.width_px, self.height_px), sharex='col')
            fig.suptitle("Individual CQR Training History")

            for i, quantile in enumerate(quantiles):
                training_history = self.train_history_list[i]
                epochs = training_history['epochs']

                # Learning Rate (left column)
                ax_lr = axs[i, 0]
                ax_lr.plot(epochs, training_history['learning_rate'], color='red')
                ax_lr.set_yscale('log')
                ax_lr.set_ylabel(f'CQR (q={quantile})\nLearning Rate')
                ax_lr.set_xlabel('Epochs')

                # Losses (right column) with twin y-axis
                ax_loss = axs[i, 1]
                ax_val = ax_loss.twinx()
                ax_loss.plot(epochs, training_history['training_loss'], color='green', label='Training Loss')
                ax_val.plot(epochs, training_history['validation_loss'], color='blue', label='Validation Loss')

                ax_loss.set_yscale('log')
                ax_val.set_yscale('log')

                ax_loss.set_ylabel(f'CQR (q={quantile})\nTraining Loss', color='green')
                ax_val.set_ylabel(f'CQR (q={quantile})\nValidation Loss', color='blue')
                ax_loss.set_xlabel('Epochs')

                if i == 0:
                    ax_loss.legend(loc='upper left')
                    ax_val.legend(loc='upper right')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

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

        # setting default device
        self._set_device(torch_device=self.cqr_model.torch_device)

        # narx_input = self.input_preprocessing(states=order_states, inputs=order_inputs)
        X_narx = torch.tensor(x_test.to_numpy(), dtype=self.dtype)

        # making prediction
        with torch.no_grad():
            Y_pred = self.cqr_model(X_narx).cpu().numpy().T

        # setting up plots
        #fig, ax = plt.subplots(n_x, figsize=(24, 6 * n_x))
        #fig.suptitle('QR Error plots')

        fig = make_subplots(rows=n_x, cols=1, shared_xaxes=True)
        fig.update_layout(height=self.height_px * 100, width=self.width_px * 100, title_text="QR Error Plots",
                          showlegend=True)

        # sorting with timestamps
        x = t_test.reshape(-1, )
        sorted_indices = np.argsort(x)  # Get indices that would sort x
        x_sorted = x[sorted_indices]

        # plot for each state
        for i in range(n_x):
            col_name = f'state_{i+1}_next'
            # plot the real mean
            #ax[i].plot(x_sorted, y_calib[i, :][sorted_indices], label=f'real mean')
            fig.add_trace(go.Scatter(x=x_sorted, y=y_test[col_name][sorted_indices],
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

        # setting default device
        self._set_device(torch_device=self.full_model.torch_device)

        # narx_input = self.input_preprocessing(states=order_states, inputs=order_inputs)
        X_torch = torch.tensor(x_test.to_numpy(), dtype=self.dtype)

        # making full model prediction
        with torch.no_grad():
            y_pred = self.full_model(X_torch)

        # doing postprocessing containing the conformalisation step
        x0, x0_cqr_high, x0_cqr_low = self._post_processing(y=y_pred)

        # Sorting according to timestamps
        x = t_test
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]

        # Create subplots
        fig = make_subplots(rows=n_x, cols=1, shared_xaxes=True)
        fig.update_layout(height=self.height_px * 100, width=self.width_px * 100, title_text="CQR State Plots",
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
                                     y=np.concatenate((x0_cqr_high[sorted_indices, i],
                                                       x0_cqr_low[sorted_indices, i][::-1])),
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
            fig.add_trace(go.Scatter(x=x_sorted, y=x0[sorted_indices, i],
                                     mode='lines', name=f'Predicted Mean',
                                     line=dict(color='blue', dash='longdashdot'),
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            # Real mean line (show legend for the first plot of each row)
            fig.add_trace(go.Scatter(x=x_sorted, y=y_test.to_numpy()[sorted_indices, i],
                                     mode='lines', name=f'Real Mean',
                                     line=dict(color='orange'),
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            # CQR High quantile (show legend for the first plot of each row)
            fig.add_trace(go.Scatter(x=x_sorted, y=x0_cqr_high[sorted_indices, i],
                                     mode='markers', name=f'High Quantile={high_quantile}',
                                     marker=dict(color='green', size=6),
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            # CQR Low quantile (show legend for the first plot of each row)
            fig.add_trace(go.Scatter(x=x_sorted, y=x0_cqr_low[sorted_indices, i],
                                     mode='markers', name=f'Low Quantile={low_quantile}',
                                     marker=dict(color='purple', size=6),
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            fig.update_yaxes(title_text=f' State {i + 1}', row=i + 1, col=1)
            fig.update_xaxes(title_text='Times Stamp [s]', row=i + 1, col=1)

        # Show plot
        fig.show()
        return None


    def make_branch_old(self, u0_traj):
        assert self.flags['qr_ready'], "Quantile regressor not ready."
        assert self.flags['cqr_ready'], "Quantile regressor not conformalised."
        assert self.flags['cqr_initial_condition_ready'], "CQR not initialised"
        assert u0_traj.shape[1] == self.n_u, \
            f"u0 should have have {self.n_u} columns but instead found {u0_traj.shape[1]}!"

        # storage for later retrieval
        n_x = self.n_x
        n_u = self.n_u
        order = self.order
        state_n = self.states.reshape((1, -1))
        if self.order>1:
            input_n = self.inputs.reshape((1, -1))
        alpha_branch = [1]
        time_branch = [0.0]
        steps = u0_traj.shape[0]
        states_branch = [self.states[0, :].reshape(1, -1)]

        # generating the branches
        for i in range(steps):

            # init
            u0 = u0_traj[i,:].reshape((1, -1))
            n_samples = state_n.shape[0]

            # segregating states and inputs
            u0_stacked = np.vstack([u0] * n_samples)

            # stacking all data
            if self.order > 1:
                X = np.hstack([state_n, u0_stacked, input_n])
            else:
                X = np.hstack([state_n, u0_stacked])

            # setting default device
            self._set_device(torch_device=self.full_model.torch_device)

            # narx_input = self.input_preprocessing(states=order_states, inputs=order_inputs)
            X_torch = torch.tensor(X, dtype=self.dtype)

            # making full model prediction
            with torch.no_grad():
                y_pred = self.full_model(X_torch)

            # doing postprocessing containing the conformalisation step
            x0, x0_cqr_high, x0_cqr_low = self._post_processing(y=y_pred)

            # finding the upper limit
            states_3d = np.stack([x0, x0_cqr_high, x0_cqr_low], axis=0)

            # finding the limits per row and col
            max_states = np.max(states_3d, axis=0)
            min_states = np.min(states_3d, axis=0)

            # sanity check
            assert np.all(max_states>=min_states), ("Some values of the in the max_states in < min_states, "
                                                    "which is not expected. Should not happen if "
                                                    "the system is monotonic.")

            # generating random points between the max and the min
            random_states = np.random.uniform(
                low=min_states,
                high=max_states,
                size=(self.rnd_samples, *x0.shape)
            )

            random_states_2d = random_states.reshape((-1, self.n_x))

            # stacking outputs
            x0_next = np.vstack([x0, x0_cqr_high, x0_cqr_low, random_states_2d])

            # preparing the next initial conditions
            state_n = np.hstack([x0_next, np.vstack([state_n]*(3 + self.rnd_samples))])[:, 0:n_x*order]
            if self.order > 1:
                input_n = np.hstack([np.vstack([u0]*x0_next.shape[0]), np.vstack([input_n]*(3 + self.rnd_samples))])[:, 0:n_u*(order-1)]

            # stores the branched states
            if self.confidence_cutoff == 1:

                # branches not stored if confidence_cutoff = 1, equivalent to nominal mpc
                states_branch.append(np.vstack([x0]))
            else:
                states_branch.append(x0_next)
            alpha_branch.append(alpha_branch[-1]*(1-self.alpha))
            time_branch.append(time_branch[-1] + self.t_step)

            # force cutoff is confidence is low
            if alpha_branch[-1] < self.confidence_cutoff:
                break

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
        fig.update_layout(height=self.height_px * 100 * (n_x+n_u), width=self.width_px * 100, title_text="CQR State Branch Plots",
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
                        y=[min(states[j][:, i]), min(states[j+1][:, i]), max(states[j+1][:, i]), max(states[j][:, i])],
                        fill="toself",
                        fillcolor=f"rgba(255, 255, 0, {alphas[j]})",  # Grey shaded region
                        line=dict(color="rgba(255,255,255,0)"),  # No border
                        name=f"Confidence={alphas[j]}", showlegend=False),

                        row=i + 1, col=1)


                fig.add_trace(go.Scatter(x=[t]*states[j][:,i].shape[0], y=states[j][:, i],
                                         mode='markers', name=f'Branches',
                                         marker=dict(color='pink', size=2),
                                         showlegend=True if i==0 and j==0 else False),
                              row=i + 1, col=1)

                # extracting mean prediction
                mean_prediction.append(states[j][0, i])

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
            fig.add_trace(go.Scatter(x=time_stamps, y=u0_traj[:, j],
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

    def plot_branch_matplotlib_old(self, t0=0.0, show_plot=True):
        n_x = self.n_x
        n_u = self.n_u
        time_stamp_states = [num + t0 for num in self.branches['time_stamps']]
        states = self.branches['states']
        alphas = self.branches['alphas']
        u0_traj = self.branches['u0_traj']
        time_stamp_inputs = np.arange(t0, t0 + (self.t_step * u0_traj.shape[0]), self.t_step)[0:u0_traj.shape[0]]

        # Create subplots
        fig, axes = plt.subplots(n_x + n_u, 1, figsize=(self.width_px, self.height_px), sharex=True)

        if n_x + n_u == 1:  # If there's only one subplot, wrap axes in a list for consistency
            axes = [axes]

        # Loop through each state
        for i in range(n_x):
            ax = axes[i]
            mean_prediction = []

            for j, t in enumerate(time_stamp_states):
                if j < len(time_stamp_states) - 1:
                    # Add shaded region for confidence
                    ax.fill_between(
                        [time_stamp_states[j], time_stamp_states[j + 1]],
                        [min(states[j][:, i]), min(states[j + 1][:, i])],
                        [max(states[j][:, i]), max(states[j + 1][:, i])],
                        color='yellow',
                        alpha=alphas[j],
                        label=f'Confidence' if j == 0 and i==0 else None,
                    )

                # Scatter plot of branches
                ax.scatter([t] * states[j][:, i].shape[0], states[j][:, i], color='pink', s=2,
                           label='Branches' if i == 0 and j == 0 else None)

                # Extracting mean prediction
                mean_prediction.append(states[j][0, i])

            # Line plot of mean prediction
            ax.plot(time_stamp_states, mean_prediction,
                    linestyle='dashed', color='red', label='Nominal Projection' if i == 0 else None)

            ax.set_ylabel(f'State {i + 1}')
            #ax.legend()

        for j in range(n_u):
            ax = axes[n_x + j]

            # Line plot of input trajectory
            ax.plot(time_stamp_inputs, u0_traj[:, j], linestyle='dashed', color='red', label='MPC trajectory')

            ax.set_ylabel(f'Inputs {j + 1}')
            #ax.legend()

        # Set x-axis labels on all plots after looping through inputs and states.
        for ax in axes:
            ax.set_xlabel('Times [s]')
            ax.grid()

        fig.suptitle("CQR State Branch Plots", fontsize=16)

        # Show or return the plot based on `show_plot`
        if show_plot:
            fig.show()
        else:
            plt.close(fig)
            return fig, axes

