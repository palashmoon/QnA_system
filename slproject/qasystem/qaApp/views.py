from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import sys
import os
import psycopg2

# Create your views here.
currentdir = os.getcwd()
sys.path.insert(0, os.path.join(currentdir, "../"))

from preprocessing import Preprocessing
preprocessing = Preprocessing()
preprocessing.start()

from test import returnAnswer
from predict_bert import return_bert_answer



question = "How much fees do we have to pay for using LaTeX?"
context = "LaTeX is a high-quality typesetting system; it includes features designed for the production of technical and scientific documentation. LaTeX is the de facto standard for the communication and publication of scientific documents. LaTeX is available as free software. You don't have to pay for using LaTeX, i.e., there are no license fees, etc. But you are, of course, invited to support the maintenance and development efforts through a donation to the TeX Users Group (choose LaTeX Project contribution) if you are satisfied with LaTeX. You can also sponsor the work of LaTeX team members through the GitHub sponsor program at the moment for Frank, David and Joseph. Your contribution will be matched by GitHub in the first year and goes 100% to the developers. The volunteer efforts that provide you with LaTeX need financial support, so thanks for any contribution you are willing to make."

@csrf_exempt
def index(request):
    context = {'a': 1}
    return render(request, 'index.html', context)


@csrf_exempt
def predictAnswer(request):
    dict = request.POST.dict()
    context = dict['input']
    question = dict['question']

    predicted_answer = returnAnswer(context, question)
    bert_answer = return_bert_answer(context, question)
    context = "Paragraph is : " + context
    question = "Question is : " + question
    predicted_answer = "Answer is : " + predicted_answer
    bert_answer = "Bert Answer : " + bert_answer
    
    # connect with database and commit
    conn = psycopg2.connect(
    database="mydb",
    user='postgres1',
    password='root',
    host='localhost',
    port='5432')

    cursor = conn.cursor()

    cursor.execute("INSERT into QnA(question, answer) VALUES (%s , %s)", (question, predicted_answer))
    conn.commit()

    returndict = {'paragraph':context,'question':question,'answer': predicted_answer, 'bertanswer': bert_answer}    
    return render(request, 'index.html', returndict)
