import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

currentdir = os.getcwd()
df = pd.read_csv(os.path.join(currentdir, "data/new.csv"))
df = df.drop(["Unnamed: 0"], axis = 1)

def plot_answer_start_analysis():
    start_indices = np.zeros(518)
    for ele in df.answer_start:
        start_indices[ele] += 1

    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(20,8))
    plt.bar(np.arange(0,201),start_indices[:201], color ='darkorchid', width = 0.8)

    plt.xlabel("Index of context word")
    plt.ylabel("Number of examples")
    plt.title("Answer start indices")
    plt.savefig(os.path.join(currentdir, "qaApp/static/answer_start_indices.png"))

def plot_answer_length_analysis():
    lengths = np.zeros(44)
    for ele in df.answer_length:
        lengths[ele] += 1

    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(20,8))
    plt.bar(np.arange(1,21),lengths[1:21], color ='darkorchid', width = 0.8)

    plt.xlabel("Index of context word")
    plt.ylabel("Number of examples")
    plt.title("Answer length ")
    plt.savefig(os.path.join(currentdir, "qaApp/static/answer_lengths.png"))

def plot_wordcloud():
    comment_words = ''
    stopwords = set(STOPWORDS)
    
    for val in df.context:
        val = str(val)
        tokens = val.split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens)+" "
    wordcloud = WordCloud(width = 800, height = 800,background_color ='white',stopwords = stopwords,min_font_size = 10).generate(comment_words)
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig(os.path.join(currentdir, "qaApp/static/wordcloud.png"))
