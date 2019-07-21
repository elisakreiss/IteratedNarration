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

# TODO: put model and data wrangling,... in separate files
# TODO: save csv files un runs folder too
# TODO: fix visualizations
# TODO: also save which data was dev and test

class LAModel(nn.Module):

    def __init__(self, hidden_dim):
        super(LAModel, self).__init__()
        # initialize pretrained bert model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # input: word embeddings; output: hidden states (dim = hidden_dim)
        self.lstm = nn.LSTM(768, hidden_dim, bidirectional=True)

        # initialize attention module
        self.attention = Attention(hidden_dim)

        # initialize layer for linear mapping
        self.linear = nn.Linear(hidden_dim*2, 1)

    def forward(self, sentence, post_computation_fct=None):
        # retrieve bert embeddings
        encoded_layers, _ = self.bert(sentence, output_all_encoded_layers=False)

        # lstm_out contains all hidden layers
        lstm_out, _ = self.lstm(encoded_layers[0].view(len(sentence[0]), 1, -1))

        attention_out = self.attention(sentence, lstm_out)

        # linear layer on top of last hidden layer (ensures value between 0 and 1)
        prediction = torch.sigmoid(self.linear(attention_out))

        if post_computation_fct is not None:
            post_computation_fct(lstm_out, prediction)

        return prediction

class Attention(nn.Module):

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention1 = nn.Linear(hidden_dim*2, 50)
        self.attention2 = nn.Linear(50, 1)

    def forward(self, sentence, lstm_out):
        attention1 = torch.tanh(self.attention1(lstm_out))
        attention2 = torch.softmax(self.attention2(attention1), dim=0)
        # transform attention2 dimensionality
        transformed_att2 = torch.transpose(attention2.view(len(sentence[0]), -1), 0, 1)
        # matrix multiplication of second attention layer and lstm_out
        dot_product = torch.mm(transformed_att2, lstm_out.view(len(sentence[0]), -1))

        return dot_product


# def post_computation_fct(lstm_out, attention1, attention2, prediction):
#     print('some visualizations can be done here')

# Print iterations progress
def print_progress_bar(iteration, total, decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    progress_bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % ('Progress: ', progress_bar, percent, 'Complete'), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def import_data(ling_measure, data_type):
    df_training = pandas.read_csv("./data/data_prep/"+ling_measure+"_"+data_type+".csv")
    data = df_training.values.tolist()
    random.shuffle(data)
    return data

def split_and_tokenize(data, cv_fold):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)

    split_data = [[] for i in range(cv_fold)]
    for data_id, story_target_val in enumerate(data):
        bucket = math.floor(data_id/(len(data)/cv_fold))
        tokenized_text = tokenizer.tokenize(story_target_val[0])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # each data point is a 4-level dictionary:
        # 1) story 2) tokenized story 3) story with BERT indeces 4) target label
        data_dict = {
            "story": story_target_val[0],
            "story_tokenized": tokenized_text,
            "story_indexed": torch.tensor([indexed_tokens]),
            "target_label": torch.tensor(story_target_val[1])
        }
        split_data[bucket].append(data_dict)
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
    for training_sample in training_data:
        targets.append(training_sample['target_label'])
    return torch.tensor(np.mean(targets))

def save_weights(model, out_dir, cv_id, epoch):
    filename = "epoch_" + str(epoch)
    path = os.path.join(out_dir, "model_weights", cv_id, filename)
    torch.save(model.state_dict(), path + ".pt")

def evaluate(model, data, loss_function, file_id, out_dir, epoch=0, baseline_data=None):
    csv_data = [['DataType', 'Epoch', 'Prediction', 'Target', 'Loss']]

    with torch.no_grad():
        for training_sample in data:
            # run forward pass
            prediction = model(training_sample['story_indexed'])

            # compute loss and write to csv
            devdata_loss = loss_function(prediction, training_sample['target_label'])
            csv_data.append(['testing', epoch, prediction.detach().numpy()[0][0], training_sample['target_label'].detach().numpy(), devdata_loss.detach().numpy()])

            if baseline_data is not None:
                baseline_loss = loss_function(baseline_data, training_sample['target_label'])
                csv_data.append(['baseline', epoch, baseline_data.numpy(), training_sample['target_label'].detach().numpy(), baseline_loss.detach().numpy()])

        # with open('losses_' + file_id + '.csv', 'w') as csv_file:
        #     writer = csv.writer(csv_file)
        #     writer.writerows(csv_data)

        # csv_file.close()

def train_model(model, data, config, out_dir, cv_id, save_csv=True):

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    csv_trainingdata = [['DataType', 'Epoch', 'Prediction', 'Target', 'Loss']]

    for epoch in range(config['num_epochs']):
        print("epoch: " + str(epoch + 1))
        count = 0
        for training_sample in data['training_data']:
            count += 1
            print_progress_bar(count, len(data['training_data']), length=50)

            # pytorch accumulates gradients
            # we need to clear them out before each instance
            model.zero_grad()

            # run forward pass
            prediction = model(training_sample['story_indexed'])

            # compute loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(prediction, training_sample['target_label'])
            loss.backward()
            optimizer.step()

            # save losses for visualization
            csv_trainingdata.append(['training', epoch, prediction.detach().numpy()[0][0], training_sample['target_label'].detach().numpy(), loss.detach().numpy()])

        # save training loss in csv
        if save_csv:
            print("writing csv")
            # with open('losses_' + cv_id + '.csv', 'w') as csv_file:
            #     writer = csv.writer(csv_file)
            #     writer.writerows(csv_trainingdata)

            # csv_file.close()

        # save model weights
        save_weights(model, out_dir, cv_id, epoch)

        # model evaluation on dev_data for current epoch
        evaluate(model, data['dev_data'], loss_function, cv_id, epoch, data['baseline_data'], out_dir)

    return model

def main():
    torch.manual_seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    out_dir = args.out_dir
    config_file_path = os.path.join(out_dir, "config.json")
    config = json.load(open(config_file_path, "r"))

    # MODEL
    model = LAModel(config['hidden_dim'])
    os.mkdir(os.path.join(out_dir, "model_weights"))

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
        os.mkdir(os.path.join(out_dir, "model_weights", training_id))
        train_model(model, data, config, out_dir, training_id)

main()
