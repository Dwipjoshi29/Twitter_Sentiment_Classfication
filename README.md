# Twitter_Sentiment_Classfication
TWITTER SENTIMENT CLASSIFICATION :
This project demonstrates the creation of a binary classification recurrent neural network (RNN) model using Keras for text classification tasks.

Introduction
This repository contains Python code to create a binary classification RNN model for classifying text data into two classes. The model is designed to process variable-length sequences of text data and make binary predictions.

Code Overview
The main components of the code are as follows:

Model Creation: The create_bin_class_rnn() function initializes and configures a binary classification RNN model using Keras. The model consists of an input layer, an embedding layer, an LSTM layer, and a dense output layer with a sigmoid activation function.

Model Compilation: The compile() method is used to compile the model with the binary cross-entropy loss function and evaluation metrics such as accuracy, precision, and recall.

Training the Model: The fit() method is used to train the model on training data. Training is performed for a specified number of epochs with optional validation data.

Model Evaluation: After training, the model can be evaluated on a separate test dataset using the evaluate() method.

Usage :-
Requirements: Ensure that you have the necessary Python libraries installed, including TensorFlow, Keras, and any additional dependencies required by your environment.

Data Preparation: Prepare your text data for training and testing. Ensure that the data is preprocessed and formatted appropriately for input to the RNN model.

Model Configuration: Modify the parameters of the create_bin_class_rnn() function as needed to customize the model architecture and hyperparameters.

Training: Use the fit() method to train the model on your training data. Provide training data, validation data (optional), and specify the number of epochs for training.

Evaluation: After training, evaluate the model's performance on a separate test dataset using the evaluate() method.

Example Usage

# Determine max_vocabulary_size and embedding_output_dim
max_vocabulary_size = len(loaded_vectorization_layer.get_vocabulary())
embedding_output_dim = 50

# Create binary classification RNN model
rnn_model = create_bin_class_rnn(max_vocabulary_size, embedding_output_dim, max_sequence_len)

# Compile the model
rnn_model.compile(loss="binary_crossentropy",
                   metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Train the model
rnn_model.fit(training_data_gen,
              epochs=num_epochs,
              validation_data=cv_data_gen,
              steps_per_epoch=200,
              validation_steps=9)

# Evaluate the model
test_loss, test_accuracy, test_precision, test_recall = rnn_model.evaluate(test_data_gen)
