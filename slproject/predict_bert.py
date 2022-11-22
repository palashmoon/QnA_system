import torch
import re
from transformers import BertForQuestionAnswering, BertTokenizer

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def return_bert_answer(answer_text, question):
    input_tokenized = tokenizer.encode(question, answer_text)
    input_token_list = tokenizer.convert_ids_to_tokens(input_tokenized)
    pos_of_sep = input_tokenized.index(tokenizer.sep_token_id)

    num_context = pos_of_sep + 1
    num_question = len(input_tokenized) - num_context

    id_list = [0 for i in range(num_context)] + [1 for i in range(num_question)]
    outputs = model(torch.tensor([input_tokenized]), token_type_ids=torch.tensor([id_list]), return_dict=True) 

    predicted_start = outputs.start_logits
    predicted_end = outputs.end_logits
    answer_start = torch.argmax(predicted_start)
    answer_end = torch.argmax(predicted_end)

    answer = ' '.join(input_token_list[answer_start:answer_end+1])
    final_answer = ''
    i=0
    final_answer = re.sub(" ##", "", answer)
    return final_answer