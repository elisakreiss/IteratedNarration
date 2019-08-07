import csv
import random
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertModel, BertTokenizer

torch.manual_seed(1)


# BERT
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)

df_testing = pandas.read_csv("./data/data_prep/noevstories_fewerhedges.csv")

data = df_testing.values.tolist()
random.shuffle(data)

run_id="testing"

testing_data = []
csvDataStory = [['story_tokenized', 'tensor', 'story_id', 'story', 'token_id', 'story_cond', 'hedges']]
for data_id in range(len(data)):
    tokenized_text = tokenizer.tokenize(data[data_id][2])
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    testing_data.append(torch.tensor([indexed_tokens]))

    # csvFileCreation
    for token_id in range(len(tokenized_text)):
        csvDataStory.append([tokenized_text[token_id], indexed_tokens[token_id], data_id, data[data_id][0], token_id, data[data_id][0], data[data_id][1]])

with open('model_visualization/noev_hedges/tokenized_stories.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvDataStory)
    csvFile.close()


# hidden dimensions
# HIDDEN_DIM = 350
HIDDEN_DIM = 200

class LSTM_Forward(nn.Module):

    def __init__(self, hidden_dim):
        super(LSTM_Forward, self).__init__()
        self.hidden_dim = hidden_dim

        # self.word_embeddings, num_embeddings, embedding_dim = create_emb_layer(wordembedding_matrix, True)
        self.Bert = BertModel.from_pretrained('bert-base-uncased')

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.lstm = nn.LSTM(768, hidden_dim, bidirectional=True)

        # this only works for sentences of length 4
        self.attention1 = nn.Linear(hidden_dim*2,50)
        self.attention2 = nn.Linear(50,1)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_dim*2, 1)

    def forward(self, sentence, sen_id):
        # embeds = self.word_embeddings(sentence)
        encoded_layers, _ = self.Bert(sentence,output_all_encoded_layers=False)

        # lstm_out contains all hidden layers
        lstm_out, _ = self.lstm(encoded_layers[0].view(len(sentence[0]), 1, -1))

        attention1 = torch.tanh(self.attention1(lstm_out))
        attention2 = torch.softmax(self.attention2(attention1),dim=0)
        # CSVFILECREATION
        for token_id in range(len(attention2.squeeze().tolist())):
            csvDataTensor.append([sentence.squeeze().tolist()[token_id], attention2.squeeze().tolist()[token_id], sen_id, token_id])

        dot_product = torch.mm(torch.transpose(attention2.view(len(sentence[0]), -1),0,1),lstm_out.view(len(sentence[0]), -1))

        # linear layer on top of last hidden layer
        prediction = torch.sigmoid(self.linear(dot_product))
        print('prediction: ', prediction.shape)

        return prediction

model = LSTM_Forward(HIDDEN_DIM)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

model.load_state_dict(torch.load("data/model_weights/suspect_guilt_cv0.pt"))
model.eval()
print("loaded weights")


with torch.no_grad():
    losses = []
    # csvData = [['Data_type', 'Prediction', 'Target', 'Loss']]
    csvDataTensor = [['tensor', 'attention', 'story_id', 'token_id']]
    csvDataLoss = [['story_id', 'prediction']]
    sen_id = 0 # for csv file creation
    for sentence in testing_data:
        # transform sentence into tensor of embeddings
        # sentence_in = prepare_sequence(sentence.split(), w_embed_dict)
        sentence_in = sentence
        # look up target "embedding"
        # targets = prepare_sequence(target, tag_to_ix)
        # targets = torch.tensor(target)
        # print("target: ", target)

        # Step 3. Run our forward pass.
        print(sentence_in)
        prediction = model(sentence_in, sen_id)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # loss = loss_function(prediction, targets)
        # losses.append(loss)
        # csvData.append(['final_testing', prediction.detach().numpy()[0][0], targets.detach().numpy(), loss.detach().numpy()])
        csvDataLoss.append([sen_id, prediction.detach().numpy()[0][0]])

        # with open('losses_'+ling_measure+'_'+run_id+'.csv', 'w') as csvFile:
        #     writer = csv.writer(csvFile)
        #     writer.writerows(csvData)

        # csvFile.close()

        sen_id += 1
    
    with open('model_visualization/noev_hedges/attention.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvDataTensor)

    csvFile.close()

    with open('model_visualization/noev_hedges/loss.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvDataLoss)

    csvFile.close()








