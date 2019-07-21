import json
import os
import argparse
import csv
import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertModel, BertTokenizer
import pandas

# TODO put model and data wrangling,... in separate files

class LAModel(nn.Module):

    def __init__(self, hidden_dim):
        super(LAModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(768, hidden_dim, bidirectional=True)

        # TODO: write attention module
        self.attention1 = nn.Linear(hidden_dim*2, 50)
        self.attention2 = nn.Linear(50, 1)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_dim*2, 1)

    def forward(self, sentence, post_computation_fct=None):
        encoded_layers, _ = self.bert(sentence, output_all_encoded_layers=False)

        # lstm_out contains all hidden layers
        lstm_out, _ = self.lstm(encoded_layers[0].view(len(sentence[0]), 1, -1))

        attention1 = torch.tanh(self.attention1(lstm_out))

        attention2 = torch.softmax(self.attention2(attention1), dim=0)

        dot_product = torch.mm(torch.transpose(attention2.view(len(sentence[0]), -1), 0, 1), lstm_out.view(len(sentence[0]), -1))

        # linear layer on top of last hidden layer
        prediction = torch.sigmoid(self.linear(dot_product))

        if post_computation_fct is not None:
            post_computation_fct(lstm_out, attention1, attention2, prediction)

        return prediction

def post_computation_fct(lstm_out, attention1, attention2, prediction):
    print('some visualizations can be done here')

# Print iterations progress
def print_progress_bar(iteration, total, prefix='Progress: ', suffix='Complete', decimals=1, length=100, fill='█'):
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
    # TODO: include data path in config file and as input variable
    df_training = pandas.read_csv("./data/data_prep/"+ling_measure+"_"+data_type+".csv")
    data = df_training.values.tolist()
    random.shuffle(data)
    return data

def split_and_tokenize(data, cv_fold):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)
    print("tokenizer is loaded")

    split_data = [[] for i in range(cv_fold)]
    for data_id, story_target_val in enumerate(data):
        bucket = math.floor(data_id/(len(data)/cv_fold))
        tokenized_text = tokenizer.tokenize(story_target_val[0])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # TODO: make this into a t-level dictionary:
        # 1) tokenized story 2) indeces 3) target label
        split_data[bucket].append([torch.tensor([indexed_tokens]), torch.tensor(story_target_val[1])])
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

def evaluate(model, data, loss_function, file_id, epoch=0, baseline_data=None):
    csv_data = [['DataType', 'Epoch', 'Prediction', 'Target', 'Loss']]

    with torch.no_grad():
        for sentence, target in data:
            # run forward pass
            prediction = model(sentence)

            # compute loss and write to csv
            devdata_loss = loss_function(prediction, target)
            csv_data.append(['testing', epoch, prediction.detach().numpy()[0][0], target.detach().numpy(), devdata_loss.detach().numpy()])

            if baseline_data is not None:
                baseline_loss = loss_function(baseline_data, target)
                csv_data.append(['baseline', epoch, baseline_data.numpy(), target.detach().numpy(), baseline_loss.detach().numpy()])

        # with open('losses_' + file_id + '.csv', 'w') as csv_file:
        #     writer = csv.writer(csv_file)
        #     writer.writerows(csv_data)

        # csv_file.close()

def train_model(model, data, config, cv_id, save_csv=True):

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    csv_trainingdata = [['DataType', 'Epoch', 'Prediction', 'Target', 'Loss']]

    for epoch in range(config['num_epochs']):
        print("epoch: " + str(epoch + 1))
        count = 0
        for sentence, target in data['training_data']:
            count += 1
            print_progress_bar(count, len(data['training_data']), length=50)

            # pytorch accumulates gradients
            # we need to clear them out before each instance
            model.zero_grad()

            # run forward pass
            prediction = model(sentence)

            # compute loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(prediction, target)
            loss.backward()
            optimizer.step()

            # save losses for visualization
            csv_trainingdata.append(['training', epoch, prediction.detach().numpy()[0][0], target.detach().numpy(), loss.detach().numpy()])

        if save_csv:
            print("writing csv")
            # with open('losses_' + cv_id + '.csv', 'w') as csv_file:
            #     writer = csv.writer(csv_file)
            #     writer.writerows(csv_trainingdata)

            # csv_file.close()

        # model evaluation on dev_data for current epoch
        evaluate(model, data['dev_data'], loss_function, cv_id, epoch, data['baseline_data'])

    return model

def main():
    torch.manual_seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    # parser.add_argument("--run", required=True)
    # parser.add_argument("--subject", required=False, default=None, type=int)

    args = parser.parse_args()

    out_dir = args.out_dir
    config_file_path = os.path.join(out_dir, "config.json")
    config = json.load(open(config_file_path, "r"))

    # MODEL
    model = LAModel(config['hidden_dim'])

    # DATA
    # import (training) data
    raw_data = import_data(config['ling_measure'], 'training')
    # split data according to cross validation parameter & tokenize
    formatted_data = split_and_tokenize(data=raw_data, cv_fold=config['cv_fold'])

    for cv_fold_id, dev_data in enumerate(formatted_data):
        print("")
        # define id for cross validation step
        training_id = config['ling_measure'] + '_cv' + str(cv_fold_id)
        print(training_id)

        # DATA PREPARATION
        # prepare training data
        training_data = create_trainingbucket(formatted_data, cv_fold_id)
        # compute mean target for training data for baseline
        baseline_data = target_mean(training_data)
        data = {
            'training_data': training_data,
            'dev_data': dev_data,
            'baseline_data': baseline_data
        }

        # TRAINING
        trained_model = train_model(model, data, config, training_id)

        # save final model weights
        # TODO: save model_weights in between as well
        # TODO: also save which data was dev and test
        # torch.save(trained_model.state_dict(), "model_weights/"+training_id+".pt")

main()
