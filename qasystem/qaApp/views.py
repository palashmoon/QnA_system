from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create your views here.
df = pd.read_csv("/home/vishal/Documents/SL/testproject/qasystem/data/new.csv")
df = df.drop(["Unnamed: 0"], axis = 1)

question = "How much fees do we have to pay for using LaTeX?"
context = "LaTeX is a high-quality typesetting system; it includes features designed for the production of technical and scientific documentation. LaTeX is the de facto standard for the communication and publication of scientific documents. LaTeX is available as free software. You don't have to pay for using LaTeX, i.e., there are no license fees, etc. But you are, of course, invited to support the maintenance and development efforts through a donation to the TeX Users Group (choose LaTeX Project contribution) if you are satisfied with LaTeX. You can also sponsor the work of LaTeX team members through the GitHub sponsor program at the moment for Frank, David and Joseph. Your contribution will be matched by GitHub in the first year and goes 100% to the developers. The volunteer efforts that provide you with LaTeX need financial support, so thanks for any contribution you are willing to make."

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

model = tf.keras.models.load_model("/home/vishal/Documents/SL/testproject/qasystem/data/model(2).h5")

def plot_most_likely_answers(outputs, context_list):
    index_array = np.argsort(outputs)
    predicted_answer_list = []
    probabilities = []
    for i in range(-1,-21, -1):
        if(len(context_list) <= index_array[i]):
            predicted_answer_list.append("None")
        else:
            predicted_answer_list.append(context_list[index_array[i]])
        probabilities.append(outputs[index_array[i]] * 100)
    plt.figure(figsize=(20,8))
    plt.bar(predicted_answer_list, probabilities, color ='maroon', width = 0.4)
 
    plt.xlabel("Predicted Answers")
    plt.ylabel("Probalility of answer")
    plt.title("Maximum likely answers")
    plt.savefig("answer_list.png")

    


def convert_to_tokens(inp, word_dict, length):
        tokens_list = []
        lis = inp.lower().split()

        for l in lis:
            if(l in word_dict):
              tokens_list.append(word_dict[l])
            else:
              tokens_list.append(0)
        tokenized_inp = tf.keras.utils.pad_sequences([tokens_list], maxlen=length, padding='post')
        return tokenized_inp

dimensionality = 256

context_input = tf.keras.layers.Input(shape=(None, ))
first_embedding_layer = model.get_layer("embedding")
context_embedding = first_embedding_layer(context_input)
first_lstm_layer = model.get_layer("lstm")
context_outputs, state_h, state_c = first_lstm_layer(context_embedding)
context_states = [state_h, state_c]

question_input = tf.keras.layers.Input(shape=(None,))
second_embedding_layer = model.get_layer("embedding_1")
question_embedding = second_embedding_layer(question_input)
second_lstm_layer = model.get_layer("lstm_1")
question_outputs, _, _ = second_lstm_layer(question_embedding, initial_state=context_states)

middle_dense = model.get_layer("dense")
dense_output = middle_dense(question_outputs)
last_dense = model.get_layer("dense_1")
final_output = last_dense(dense_output)

encoder_model = tf.keras.models.Model(context_input, context_states)
decoder_state_input_h = tf.keras.layers.Input(shape=(dimensionality,))
decoder_state_input_c = tf.keras.layers.Input(shape=(dimensionality,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = second_lstm_layer(question_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = middle_dense(decoder_outputs)
decoder_outputs2 = last_dense(decoder_outputs)

decoder_model = tf.keras.models.Model(
    [question_input] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states)


@csrf_exempt
def index(request):
    context = {'a': 1}
    return render(request, 'index.html', context)


@csrf_exempt
def predictAnswer(request):
    dict = request.POST.dict()
    context = dict['input']
    question = dict['question']
    

    # Combine the tokens in the answer and print it out.
    input_context = convert_to_tokens(context, context_word_dict, max_context_len)
    states_values = encoder_model.predict(input_context)

    input_question = convert_to_tokens(question, question_word_dict, max_question_len)    
    dec_outputs, h, c = decoder_model.predict([input_question] + states_values)
    predicted_answer_index = np.argmax(dec_outputs[0, :])
    
    context_list = list(context.split())
    dec_outputs = dec_outputs.reshape(dec_outputs.shape[1])
    plot_most_likely_answers(dec_outputs, context_list)
    
    
    if(len(context_list) <= predicted_answer_index):
        predicted_answer = "None"
    else:
        predicted_answer = context_list[predicted_answer_index]

    context = {'paragraph':context,'question':question,'answer': predicted_answer}    
    return render(request, 'index.html', context)
