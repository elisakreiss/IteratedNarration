import csv
import copy
import math
import random
import pandas
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertModel, BertTokenizer

class LSTM_Forward(nn.Module):

    def __init__(self, hidden_dim):
        super(LSTM_Forward, self).__init__()
        self.hidden_dim = hidden_dim

        # self.word_embeddings, num_embeddings, embedding_dim = create_emb_layer(wordembedding_matrix, True)
        self.Bert = BertModel.from_pretrained('bert-base-uncased')


        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(768, hidden_dim, bidirectional=True)

        self.attention1 = nn.Linear(hidden_dim*2, 50)
        self.attention2 = nn.Linear(50, 1)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_dim*2, 1)

    def forward(self, sentence):
        encoded_layers, _ = self.Bert(sentence, output_all_encoded_layers=False)

        # lstm_out contains all hidden layers
        lstm_out, _ = self.lstm(encoded_layers[0].view(len(sentence[0]), 1, -1))

        attention1 = torch.tanh(self.attention1(lstm_out))

        attention2 = torch.softmax(self.attention2(attention1), dim=0)

        dot_product = torch.mm(torch.transpose(attention2.view(len(sentence[0]), -1), 0, 1), lstm_out.view(len(sentence[0]), -1))

        # linear layer on top of last hidden layer
        prediction = torch.sigmoid(self.linear(dot_product))

        return prediction

# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    progress_bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, progress_bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def import_data(ling_measure, data_type):
    df_training = pandas.read_csv("./data/data_prep/"+ling_measure+"_"+data_type+".csv")
    data = df_training.values.tolist()
    return random.shuffle(data)

def create_cv_datasplit(data, cv_fold):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)

    split_data = [[] for i in range(cv_fold)]
    for data_id, story_target_val in enumerate(data):
        bucket = math.floor(data_id/(len(data)/cv_fold))
        tokenized_text = tokenizer.tokenize(story_target_val[0])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        split_data[bucket].append([torch.tensor([indexed_tokens]), story_target_val[1]])
    
    return split_data

def create_trainingbucket(split_data, cv_fold_id):
    training_data_prep = copy.deepcopy(split_data)
    training_data_prep.pop(cv_fold_id)

    training_data = []
    for fold_list in training_data_prep:
        for training_sample in fold_list:
            training_data.append(training_sample)

    return training_data

def target_mean(training_data):
    targets = []
    for _, target in training_data:
        targets.append(target)
    return torch.tensor(np.mean(targets))

def train_model(model, training_data, dev_data, baseline_data, num_epochs, ling_measure, run_id):
    # model parameters
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    csvData = [['Data_type', 'Epoch', 'Prediction', 'Target', 'Loss']]

    for epoch in range(num_epochs):
        count = 0
        for sentence, target in training_data:
            count += 1
            print_progress_bar(count, len(training_data), prefix='Progress:', suffix='Complete', length=50)
            # pytorch accumulates gradients
            # we need to clear them out before each instance
            model.zero_grad()

            # transform sentence into tensor of embeddings
            # TODO: this should happen here!
            # sentence
            targets = torch.tensor(target)

            # run forward pass
            prediction = model(sentence)

            # compute loss, gradients, and update the parameters by 
            # calling optimizer.step()
            loss = loss_function(prediction, targets)
            loss.backward()
            optimizer.step()

            # save losses for visualization
            csvData.append(['training', epoch, prediction.detach().numpy()[0][0], targets.detach().numpy(), loss.detach().numpy()])

        # model evaluation on dev_data for current epoch
        with torch.no_grad():
            for sentence, target in dev_data:
                # transform sentence into tensor of embeddings
                # sentence
                # look up target "embedding"
                targets = torch.tensor(target)

                # run forward pass
                prediction = model(sentence)

                # compute loss and write to csv
                loss = loss_function(prediction, targets)
                csvData.append(['testing', epoch, prediction.detach().numpy()[0][0], targets.detach().numpy(), loss.detach().numpy()])

                baseline_loss = loss_function(baseline_data, targets)
                csvData.append(['baseline', epoch, baseline_data.numpy(), targets.detach().numpy(), baseline_loss.detach().numpy()])

            with open('losses_'+ling_measure+'_'+run_id+'.csv', 'w') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerows(csvData)

            csvFile.close()

    return model

def main():
    torch.manual_seed(1)

    ling_measure = "suspect_guilt"

    # MODEL
    hidden_dim = 200
    num_epochs = 30
    model = LSTM_Forward(hidden_dim)

    # DATA
    # import (training) data
    data = import_data(ling_measure, "training")
    # define k for cross validation
    cv_fold = 10
    # split data according to cross validation parameter
    # this also tokenizes -- TODO: make that cleaner
    split_data = create_cv_datasplit(data=data, cv_fold=cv_fold)

    for cv_fold_id, dev_data in enumerate(split_data):
        # prepare training data
        training_data = create_trainingbucket(split_data, cv_fold_id)
        # compute mean target for training data for baseline
        baseline_data = target_mean(training_data)

        # TRAINING
        trained_model = train_model(model, training_data, dev_data, baseline_data, num_epochs, ling_measure, cv_fold_id)

        torch.save(trained_model.state_dict(), "model_weights/"+ling_measure+'_'+cv_fold_id+".pt")

main()