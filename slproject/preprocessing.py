import os
import urllib.request
import json
import pandas as pd

class Preprocessing:
    def __init__(self):
        self.filename = 'train-v1.1.json'
        self.directory = 'squad1.1'
        self.url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
        if not os.path.exists(self.directory):
          os.makedirs(self.directory)
    
    def start(self):
        # if dataset doesnt exit then download it from the github repo
        self.download_dataset(self.url, self.filename , self.directory)
        print("Dataset is downloaded")

        # load dataset into a file
        train_data = self.load_data(self.filename, self.directory)
        print("Json file succesfully loaded")

        # prepare a csv file
        self.prepare_csvfile(train_data)
    
    def prepare_csvfile(self, train_data):
        context_list = []
        question_list = []
        answer_text_list = []
        answer_start = []
        context_data = []
        question_data = []
        answer_start_data = []
        answer_length_data = []

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
                    answer_text = question["answers"][0]["text"]
                    answer_start = question["answers"][0]["answer_start"]
                    
                    start = 0
                    for c in range(answer_start):
                      if(context[c] == ' '):
                        start += 1
                    answer_list = list(answer_text.split())
                    answer_length = len(answer_list)
                    context_data.append(context)
                    question_data.append(questions)
                    answer_start_data.append(start)
                    answer_length_data.append(answer_length)
                    q.append(questions)
                    a.append(answer_text)

            context_list.append(context)
            question_list.append(q)
            answer_text_list.append(a)
        # make a list for all and then put it    
        dict1 = {'context': context_list , 'question': question_list , 'answer': answer_text_list}
        dict2 = {'context' : context_data, 'output' : question_data, 'answer_start':answer_start_data, 'answer_length':answer_length_data}
        df2 = pd.DataFrame.from_dict(dict2)
        df2.to_csv("new1.csv")
        currentdir = os.getcwd()
        df2.to_csv(os.path.join(currentdir, "data/new.csv"))
        df1 = pd.DataFrame.from_dict(dict1)
        df1.to_csv("output.csv")

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
   
    
