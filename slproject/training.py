import pandas as pd
import numpy as np
import tensorflow as tf
import os

currentdir = os.getcwd()
print(currentdir)
model_path = os.path.join(currentdir, "data/model(2).h5")
df = pd.read_csv(os.path.join(currentdir, "data/new.csv"))
df = df.drop(["Unnamed: 0"], axis = 1)

# Preprocessing context
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df.context[0:10000])
context_tokens = tokenizer.texts_to_sequences(np.array(df.context[0:10000]))

max_context_len = max([len(i) for i in context_tokens])
context_data = tf.keras.utils.pad_sequences(context_tokens, padding='post')
context_word_dict = tokenizer.word_index
num_context_tokens = len(context_word_dict) + 1

# Preprocessing questions
tokenizer.fit_on_texts(df.output[0:10000])
question_tokens = tokenizer.texts_to_sequences(np.array(df.output[0:10000]))

max_question_len = max([len(i) for i in question_tokens])
question_data = tf.keras.utils.pad_sequences(question_tokens, padding='post')
question_word_dict = tokenizer.word_index
num_question_tokens = len(question_word_dict) + 1

# print(max_context_len, max_question_len, num_context_tokens, num_question_tokens)

answer_start = np.array(df.answer_start[0:10000], dtype=int)
answer_length = np.array(df.answer_length[0:10000], dtype=int)
onehot_start = tf.keras.utils.to_categorical(answer_start, max_context_len)
onehot_length = tf.keras.utils.to_categorical(answer_length, max_context_len)

decoder_target_data = np.array(onehot_start)
decoder_target_data.shape

dimensionality = 256
validation_data = 0.3
batch_size = 128
number_of_epochs = 10

context_input = tf.keras.layers.Input(shape=(None, ))
first_embedding_layer = tf.keras.layers.Embedding(num_context_tokens, dimensionality)
context_embedding = first_embedding_layer(context_input)
first_lstm_layer = tf.keras.layers.LSTM(dimensionality, return_state=True, dropout=0.2)
context_outputs, state_h, state_c = first_lstm_layer(context_embedding)
context_states = [state_h, state_c]

question_input = tf.keras.layers.Input(shape=(None,))
second_embedding_layer = tf.keras.layers.Embedding(num_question_tokens, dimensionality)
question_embedding = second_embedding_layer(question_input)
second_lstm_layer = tf.keras.layers.LSTM(dimensionality, return_state=True, dropout=0.2)
question_outputs, _, _ = second_lstm_layer(question_embedding, initial_state=context_states)

middle_dense = tf.keras.layers.Dense(dimensionality, activation=tf.keras.activations.relu)
dense_output = middle_dense(question_outputs)
last_dense = tf.keras.layers.Dense(max_context_len, activation=tf.keras.activations.softmax)
final_output = last_dense(dense_output)

model = tf.keras.models.Model([context_input, question_input], final_output)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

def train():
    history = model.fit([context_data, question_data],
                        np.array(onehot_start),
                        validation_split=validation_data,
                        batch_size=batch_size,
                        epochs=number_of_epochs,
                        shuffle=True)

    model.save(model_path)
