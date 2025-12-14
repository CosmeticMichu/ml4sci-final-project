
import tensorflow as tf

import tensorflow as tf

def cnn_rnn(input_shape, loss='mse', lr=1e-04, model_name="cnn_rnn"):
    '''
    CNN-RNN (Conv1D + Bidirectional LSTM) hybrid model.
    '''

    model = tf.keras.models.Sequential([
        # --- CNN Feature Extraction (Your Existing Layers) ---
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(64, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(256, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(256, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(256, 3, strides=1, activation='relu'),
        # Note: We keep the final MaxPooling1D. Its output will be a sequence, which is perfect for an RNN.
        tf.keras.layers.MaxPooling1D(2, padding='same'),
        
        # --- RNN Temporal Analysis ---
        # 1. Bidirectional LSTM: Processes the sequence in both forward and backward directions,
        #    capturing more context. 'return_sequences=False' is important here, as it
        #    outputs a single vector (the last state) instead of a sequence, which is 
        #    what a classifier (Dense layer) needs for a single prediction.
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=128, activation='tanh'),
        ),
        
        # --- Classifier/Regression Head ---
        # You can keep the dense layers for final classification/regression
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1) # Final output layer (e.g., for regression or single-class binary)
    ])
    
    model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=lr),
            loss=loss,
    )

    return model, model_name

# def cnn_rnn(input_shape, loss='mse', lr=1e-4, model_name="cnn_rnn"):
#     """
#     CNN + RNN hybrid for 1D sequence regression
#     """

#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Input(shape=input_shape),

#         # CNN feature extractor
#         tf.keras.layers.Conv1D(64, 3, activation='relu'),
#         tf.keras.layers.MaxPooling1D(2),

#         tf.keras.layers.Conv1D(128, 3, activation='relu'),
#         tf.keras.layers.MaxPooling1D(2),

#         tf.keras.layers.Conv1D(256, 3, activation='relu'),

#         # RNN over extracted temporal features
#         tf.keras.layers.Bidirectional(
#             tf.keras.layers.LSTM(128)
#         ),

#         tf.keras.layers.Dense(1024, activation='relu'),
#         tf.keras.layers.Dense(1)
#     ])

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
#         loss=loss
#     )

#     return model, model_name

import tensorflow as tf

def rnn(input_shape, loss='mse', lr=1e-04, model_name="rnn"):
    '''
    Pure Bidirectional LSTM model for sequence analysis.
    '''

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        # --- RNN Temporal Analysis Layers ---
        # 1. First Bidirectional LSTM: Set 'return_sequences=True' because we want the 
        #    output of this layer to be a sequence that is fed into the next LSTM layer.
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=128, return_sequences=True, activation='tanh')
        ),
        
        # 2. Second Bidirectional LSTM: Set 'return_sequences=False' (default behavior) 
        #    as this is the last RNN layer, and we need a single feature vector for the 
        #    Dense classification head.
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=256, activation='tanh')
        ),
        
        # --- Classifier/Regression Head (Similar to your CNN's head) ---
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1) # Final output layer
    ])
    
    model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=lr),
            loss=loss,
    )

    return model, model_name

def cnn(input_shape, loss='mse', lr=1e-04, model_name="cnn"):
    '''
    CNN model based on convolution layer + max pooling layer.
    '''

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(64, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(256, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(256, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(256, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2, padding='same'),

        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1)])
    
    model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=lr),
            loss=loss,
    )

    return model, model_name


def cnn_dropout(input_shape, loss='mse', lr=1e-04, model_name="cnn_dropout"):
    '''
    CNN model based on convolution layer + max pooling layer + dropout layer.
    '''

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(16, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(128, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1)
        ])
    
    model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=lr),
            loss=loss,
    )

    return model, model_name


