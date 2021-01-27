#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import os
import datetime
import pandas as pd
import csv
import streamlit as st
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import backend as K
#mpl.rcParams['figure.figsize'] = (8, 6)
#mpl.rcParams['axes.grid'] = False
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

def main():
  st.set_option('deprecation.showPyplotGlobalUse', False)
  st.title("Web App for Wind Power Forecasting for the Villonaco Wind Farm")
  st.sidebar.title("Model Hyperparameters")
  st.sidebar.markdown("💨")
  OUT_STEPS = st.sidebar.number_input("OUT STEPS", 1, 30, step=1, key='time_step')
  CONV_WIDTH = st.sidebar.number_input("CONV WIDTH", 1, 300, step=10, key='epoch')
  MAX_EPOCHS = st.sidebar.number_input("MAX EPOCHS", 1, 300, step=10, key='max_epoch')
  if st.sidebar.button("Train/Predict", key='train'):
    df = pd.read_csv('UGE_01_Filter.csv')
    date_time = pd.to_datetime(df.pop('DATE'), format='%Y.%m.%d %H:%M:%S')

    timestamp_s = date_time.map(datetime.datetime.timestamp)


    day = 24*60*60
    year = (365.2425)*day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    fft = tf.signal.rfft(df['POWER'])
    f_per_dataset = np.arange(0, len(fft))

    n_samples_h = len(df['POWER'])
    hours_per_year = 24*365.2524
    years_per_dataset = n_samples_h/(hours_per_year)

    f_per_year = f_per_dataset/years_per_dataset

    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n*0.8)]
    val_df = df[int(n*0.8):int(n*0.95)]
    test_df = df[int(n*0.95):]

    num_features = df.shape[1]
    

    train_df['POWER'].plot(legend=True,figsize=(16,8))
    val_df['POWER'].plot(legend=True)
    test_df['POWER'].plot(legend=True)
    


    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    

    class WindowGenerator():
      def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
          self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

      def __repr__(self):
        return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])


    def split_window(self, features):
      inputs = features[:, self.input_slice, :]
      labels = features[:, self.labels_slice, :]
      if self.label_columns is not None:
        labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
      inputs.set_shape([None, self.input_width, None])
      labels.set_shape([None, self.label_width, None])

      return inputs, labels

    WindowGenerator.split_window = split_window


    def plot(self, model=None, plot_col='POWER', max_subplots=3):
      pred = []
      inputs, labels = self.example
      plt.figure(figsize=(12, 8))
      plot_col_index = self.column_indices[plot_col]
      max_n = min(max_subplots, len(inputs))
      for n in range(max_n):
        plt.subplot(3, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
          label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
          label_col_index = plot_col_index

        if label_col_index is None:
          continue
    
        trues = labels[n, :, label_col_index].numpy()
        
        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
          predictions = model(inputs)
          plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)
          pred = predictions[n, :, label_col_index].numpy()  
        if n == 0:
          plt.legend()

      plt.xlabel('Time [h]')
      return trues, pred

    WindowGenerator.plot = plot


    def make_dataset(self, data):
      data = np.array(data, dtype=np.float32)
      ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=30,)

      ds = ds.map(self.split_window)

      return ds

    WindowGenerator.make_dataset = make_dataset

    @property
    def train(self):
      return self.make_dataset(self.train_df)

    @property
    def val(self):
      return self.make_dataset(self.val_df)

    @property
    def test(self):
      return self.make_dataset(self.test_df)

    @property
    def example(self):
      """Get and cache an example batch of `inputs, labels` for plotting."""
      result = getattr(self, '_example', None)
      if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
      return result

    WindowGenerator.train = train
    WindowGenerator.val = val
    WindowGenerator.test = test
    WindowGenerator.example = example

    multi_window = WindowGenerator(input_width=48,
                                   label_width=OUT_STEPS,
                                   shift=OUT_STEPS)

    multi_window.plot()
    
    multi_val_performance = {}
    multi_performance = {}
    
    
    st.header("Hybrid Model") 
    def compile_and_fit(model, window, lr, patience=2):
      early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience,
                                                        mode='min')
      
      def coeff_determination(y_true, y_pred):
          SS_res =  K.sum(K.square( y_true-y_pred )) 
          SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
          return ( 1 - SS_res/(SS_tot + K.epsilon()) ) 
        
      
      model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.SGD(lr=lr, momentum=0.95),
                    metrics=[tf.metrics.MeanAbsoluteError(), coeff_determination, 
                                   tf.keras.metrics.MeanAbsolutePercentageError(),
                                  tf.keras.metrics.MeanSquaredError()])

      history = model.fit(window.train, epochs=MAX_EPOCHS,
                          validation_data=window.val,
                          callbacks=[early_stopping])
      return history


    
    multi_hybrid_model = tf.keras.Sequential([
         # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(192, activation='relu', kernel_size=(CONV_WIDTH),
                              strides=1,
                              input_shape=[None, 1]),

        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(24, activation="relu"),
        
        tf.keras.layers.Dense(OUT_STEPS*num_features,
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features]),
    ])
    lr = 3e-2
    history = compile_and_fit(multi_hybrid_model, multi_window, lr)

    IPython.display.clear_output()

    multi_val_performance['Hybrid'] = multi_hybrid_model.evaluate(multi_window.val)
    multi_performance['Hybrid'] = multi_hybrid_model.evaluate(multi_window.test, verbose=0)
    trues, hybrid_pred = multi_window.plot(multi_hybrid_model)

    x = np.arange(len(multi_performance))
    width = 0.3
    metric_name = 'mean_absolute_error'
    metric_index = multi_hybrid_model.metrics_names.index('mean_absolute_error')
    val_mae = [v[metric_index] for v in multi_val_performance.values()]
    test_mae = [v[metric_index] for v in multi_performance.values()]

    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=multi_performance.keys(),
               rotation=45)
    plt.ylabel(f'MAE (average over all times and outputs)')
    st.pyplot()
    

    for name, value in multi_performance.items():
      print(f'{name:8s}: {value[1]:0.4f}, {value[2]:0.4f}, {value[3]:0.4f}, {value[4]:0.4f}')

    
    st.subheader("Coeff determination")
    metric_name = 'coeff_determination'
    metric_index = multi_hybrid_model.metrics_names.index('coeff_determination')
    val_mae = [v[metric_index] for v in multi_val_performance.values()]
    test_mae = [v[metric_index] for v in multi_performance.values()]

    fig_r2 = plt.figure(figsize=(18,6))
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=multi_performance.keys(),
               rotation=45)
    plt.ylabel(f'R2 (average over all times and outputs)')
    st.pyplot(fig_r2)
    
    st.header("Real data vs Neural Network Model Predictions [Normed]")
    fig= plt.figure(figsize=(18,6))
    plt.title('Real Data vs. Neural Network Model Predictions')
    plt.xlabel('Power Generation Day [Hour]')
    plt.ylabel('Power Generation [Normed]')
    plt.plot(hybrid_pred, label = 'Hybrid', marker = '<')
    plt.plot(trues, label = 'True', color = 'black', marker = '*', linewidth=3)
    plt.legend()
    plt.grid()
    st.pyplot(fig)

    st.header('Real Data vs. Neural Network Model Predictions [KW]' )
    fig= plt.figure(figsize=(18,6))
    plt.title('Real Data vs. Neural Network Model Predictions')
    plt.xlabel('Power Generation Day [Hour]')
    plt.ylabel('Power Generation [kW]')
    plt.plot(hybrid_pred*train_std[0]+train_mean[0], label = 'Hybrid', marker = '<')
    plt.plot(trues*train_std[0]+train_mean[0], label = 'True', color = 'black', marker = '*', linewidth=3)
    plt.legend()
    plt.grid()
    st.pyplot(fig)

    st.header("Predictions")
    st.write(hybrid_pred)


if __name__ == '__main__':
    main()

