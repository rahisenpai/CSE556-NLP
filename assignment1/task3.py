import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
from task1 import WordPieceTokenizer
from task2 import Word2VecModel
import pickle


class NeuralLMDataset(Dataset):
    def __init__(self, tokenizer, word2vec, context_size=3):
        self.pad_token = '[PAD]'
        self.tokenizer = tokenizer
        self.word2vec = word2vec
        self.context_size = context_size
        self.data = self.preprocess_data()

    def tokenize_txt_file(self, input_file: str, output_file: str) -> None:
        """Tokenize sentences from input TXT file and save results as JSON"""
        # Reading the input text file
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        results = {}
        for idx, line in enumerate(lines):
            sentence = line.strip()  # Remove leading/trailing whitespaces or newlines
            tokens = self.tokenizer.tokenize(sentence)  # Tokenize the sentence using your tokenization method
            results[str(idx)] = tokens  # Store tokens with index as the key (starting from 0)
        # Saving the results to an output JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    def preprocess_data(self):
        # self.tokenizer.construct_vocabulary("corpus.txt", vocab_size=100)
        # self.tokenize_txt_file("corpus.txt", "task3-files/tokenized_corpus.json")

        corpus = None
        with open('task2-files/tokenized_corpus.json', 'r') as f: #update to 3 before submission
            # Load the JSON data
            tokenized_corpus = json.load(f)
            
            # Convert the dictionary into a list of sentences (list of tokenized words)
            corpus = [tokens for tokens in tokenized_corpus.values()]
            
        
        self.tokenized_sentences = corpus
        # updates the word to index mapping
        self.word2idx = {word: idx for idx, word in enumerate(self.tokenizer.vocab)}
        # updates the reverse index to word mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # indexed_sentences = [[self.word2vec.embeddings(torch.tensor(self.word2idx[token], dtype=torch.long)).detach().numpy() for token in sent] for sent in self.tokenized_sentences]
        data = []
        pad_idx = self.word2idx[self.pad_token]

        # for sent in self.tokenized_sentences:
        #     if len(sent) > self.context_size:
        #         for i in range(len(sent) - self.context_size):
        #             context = [self.word2vec.embeddings(torch.tensor(self.word2idx[token], dtype=torch.long)).detach().numpy() for token in sent[i:i+self.context_size]]
        #             target = sent[i+self.context_size]
        #             target_idx = self.word2idx[target]
        #             data.append((context, target_idx))
        #     else: #pad the context with pad token
        #         context = [pad_idx] * (self.context_size - len(sent)) + [self.word2idx[token] for token in sent]
        #         context = [self.word2vec.embeddings(torch.tensor(token, dtype=torch.long)).detach().numpy() for token in context]

        for sentence in self.tokenized_sentences:
            for i in range(1, len(sentence)):
                if i < self.context_size:
                    #pad the context with pad tokens
                    context_idx = [pad_idx] * (self.context_size - i) + [self.word2idx[token] for token in sentence[:i]]
                else:
                    #enough context
                    context_idx = [self.word2idx[token] for token in sentence[i-self.context_size:i]]
                context_embed = [self.word2vec.embeddings(torch.tensor(idx, dtype=torch.long)).detach().numpy() for idx in context_idx]
                target_idx = self.word2idx[sentence[i]]
                data.append((context_embed, target_idx))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_tensor = torch.tensor(np.concatenate(context).flatten(), dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return context_tensor, target_tensor

# Define Neural Language Model Variations
class NeuralLM1(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(NeuralLM1, self).__init__()
        self.architecture = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(self, x):
        return self.architecture(x)

class NeuralLM2(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(NeuralLM2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.projection = nn.Linear(input_dim, vocab_size)
    
    def forward(self, x):
        identity = self.projection(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        out = x + identity
        return out

class NeuralLM3(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(NeuralLM3, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.relu(self.batch_norm(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)


# Accuracy and Perplexity computation
def compute_accuracy(predictions, labels):
    _, preds = torch.max(predictions, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)

def compute_perplexity(loss):
    return np.exp(loss)


# Training function with loss tracking
def train(model, train_dataloader, val_dataloader, epochs=100, lr=0.01):
    model.to("cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, train_accuracies, train_preplexity = [], [], []
    val_losses, val_accuracies, val_preplexity = [], [], []

    for epoch in range(epochs):
        #training phase
        model.train()
        train_loss, train_acc = 0, 0
        for context, target in train_dataloader:
            optimizer.zero_grad() #zero the gradients
            #forward pass
            output = model(context)
            loss = criterion(output, target)
            train_loss += loss.item()
            #backward pass and optimize
            loss.backward()
            optimizer.step()
            #compute accuracy
            train_acc += compute_accuracy(output, target)
        #calculate average training loss
        train_losses.append(train_loss / len(train_dataloader))
        train_accuracies.append(train_acc / len(train_dataloader))
        train_preplexity.append(compute_perplexity(train_losses[-1]))

        #validation phase
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for context, target in val_dataloader:
                output = model(context)
                loss = criterion(output, target)
                val_loss += loss.item()
                #compute accuracy
                val_acc += compute_accuracy(output, target)
        #calculate average validation loss
        val_losses.append(val_loss / len(val_dataloader))
        val_accuracies.append(val_acc / len(val_dataloader))
        val_preplexity.append(compute_perplexity(val_losses[-1]))

        #print losses for each epoch
        print(f"----- Epoch {epoch + 1}/{epochs} -----")
        # print(f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
        print(f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {100*train_accuracies[-1]:.2f}%, Train Perplexity: {train_preplexity[-1]:.2f}")
        print(f"Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {100*val_accuracies[-1]:.2f}%, Val Perplexity: {val_preplexity[-1]:.2f}")

    return train_losses, train_accuracies, train_preplexity, val_losses, val_accuracies, val_preplexity #return losses

# Plot loss function
def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.show()


def load_model(model_class):
    model_path = 'task2-files/final_model/final_model.pt'
    checkpoint = torch.load(model_path)
    
    model = model_class(vocab_size=checkpoint['vocab_size'], embedding_dim=checkpoint['embedding_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_loss = checkpoint['val_loss']
    val_accuracy = checkpoint['val_accuracy']
    
    return model, val_loss, val_accuracy



def predict_tokens(sentence: str, num_tokens: int, context_size: int, dataset: NeuralLMDataset, model: nn.Module):
    model.eval() #evaluation mode for predicing tokens

    tokens = dataset.tokenizer.tokenize(sentence) #tokenize the input sentence
    if len(tokens) < context_size:
        tokens = [dataset.pad_token] * (context_size - len(tokens)) + tokens #add padding tokens

    sentence_embeds = [dataset.word2vec.embeddings(torch.tensor(dataset.word2idx[token], dtype=torch.long)).detach().numpy() for token in tokens] #create embeddings for tokenized sentence

    predicted_tokens = []
    for _ in range(num_tokens):
        context = sentence_embeds[-context_size:]
        context_tensor = torch.tensor(np.concatenate(context).flatten(), dtype=torch.float32)
        output = model(context_tensor)
        predicted_idx = torch.argmax(output).item()
        predicted_token = dataset.idx2word[predicted_idx]
        predicted_tokens.append(predicted_token)
        sentence_embeds.append(dataset.word2vec.embeddings(torch.tensor(predicted_idx, dtype=torch.long)).detach().numpy())
    
    return predicted_tokens


def prediciton_pipeline(input_file, num_tokens, context_size, dataset, model):
    with open(input_file, 'r') as f:
        sentences = f.readlines()
    for sentence in sentences:
        sentence = sentence.strip()
        predicted_tokens = predict_tokens(sentence, num_tokens, context_size, dataset, model)
        print(f"Input: {sentence}")
        print(f"Predicted Tokens: {' '.join(predicted_tokens)}")
        print()



if __name__ == '__main__':
    word2vec, val_loss, val_accuracy = load_model(Word2VecModel)
    tokenizer = pickle.load(open('task1-files/tokenizer.pkl', 'rb'))

    print('imported word2vec and tokenizer')

    dataset = NeuralLMDataset(tokenizer, word2vec)

    WINDOW_SIZE = 2
    BATCH_SIZE = 1024
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.02
    TRAIN_SPLIT = 0.8

    # Split dataset into training and validation
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model1 = NeuralLM1(input_dim=30, hidden_dim=256, vocab_size=8500)
    train(model1, train_loader, val_loader, epochs=NUM_EPOCHS, lr=LEARNING_RATE)

    prediciton_pipeline('task3-files/sample_test.txt', 3, 3, dataset, model1)