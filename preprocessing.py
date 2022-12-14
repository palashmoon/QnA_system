
import os
import re
import sys
import urllib.request
import json
import pandas as pd

from yaml import load

class Preprocessing:
    def __init__(self):
        self.filename = 'train-v1.1.json'
        self.directory = 'squad1.1'
        self.url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
    
    def start(self):
        # if dataset doesnt exit then download it from the github repo
        self.download_dataset(self.url, self.filename , self.directory)
        print("Dataset is downlaoded")

        # load dataset into a file
        train_data = self.load_data(self.filename, self.directory)
        print("Json file succesfully loaded")

        # prepare a csv file
        self.prepare_csvfile(train_data)
    
    def clean_text(self, text):
        # lowercase
        x = text.lower()
        # remove puntuation
        res = re.sub('[^a-zA-Z0-9]','',x)
        res = res.strip()
        # tokenize
        return res
    
    def prepare_csvfile(self, train_data):
        context_list = []
        question_list = []
        answer_text_list = []
        answer_start = []

        print("Enter prepare csv file")
        for id in range(len(train_data["data"])):
            list_para = train_data["data"][id]["paragraphs"]
            for para in list_para:
                context = para["context"]
                qas = para["qas"]
                q = []
                a = []
                for question in qas:
                    questions = question["question"]
                    # clean question
                    questions = self.clean_text(questions)
                    # remove punctiation
                    answer_text = question["answers"][0]["text"]
                    # clean answer text
                    answer_text = self.clean_text(answer_text)
                    answer_start = question["answers"][0]["answer_start"]
                    q.append(questions)
                    a.append(answer_text)

            context_list.append(context)
            question_list.append(q)
            answer_text_list.append(a)
        print(len(context_list))
        print(len(question_list))
        print(len(answer_text_list))
        # make a list for all and then put it    
        dict = {'context': context_list , 'question': question_list , 'answer': answer_text_list}
        
        new = pd.DataFrame.from_dict(dict)
        new.to_csv("output.csv")

    def load_data(self, filename , directory):
        try:
            json_path = os.path.join(directory, filename)
            print(json_path)
            file = open(json_path)
            data = json.load(file)
            file.close()
            return data
        except:
            print("unable to load a json file")
    def download_dataset(self, url , filename, directory):
        try:
            save_path = os.path.join(directory,filename)
            
            # if not present then download
            if not os.path.exists(save_path):
                # if folder doesnt exist

                url = os.path.join(url,filename)
                urllib.request.urlretrieve(url,save_path)

        except:
            print("some error occured")

    
