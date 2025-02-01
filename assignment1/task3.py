import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from typing import List, Tuple
from task1 import WordPieceTokenizer
from task2 import Word2VecModel
import pickle


class NeuralLMDataset(Dataset):
    """
    Implements a custom dataset (similar to the one in PyTorch) to handle data preparation for
    training neural language models for next-word prediction task. This dataset is compatible
    with PyTorch's DataLoader.

    Attributes:
        pad_token: the token to use for padding the context
        tokenizer: the tokenizer to tokenize the corpus
        word2vec: the word2vec model to get word embeddings
        context_size: the size of the context window to use for next-word prediction
        tokenized_sentences: the list of tokenized sentences from the corpus
        word2idx: token to index mapping in the vocabulary
        idx2word: index to token mapping in the vocabulary
        data: list containint data samples as (context_embed, target_idx) tuples
    """

    def __init__(self, corpus_path: str, pad_token: str, vocab_size: int,
                 tokenizer: WordPieceTokenizer, word2vec: Word2VecModel, context_size: int):
        """
        Initialized the NeuralLM dataset.
        """
        self.pad_token = pad_token
        self.tokenizer = tokenizer
        self.word2vec = word2vec
        self.context_size = context_size
        self.data = self.preprocess_data(corpus_path, vocab_size)

    def tokenize_txt_file(self, input_file: str, output_file: str) -> None:
        """
        Tokenize sentences from input TXT file and save results as JSON
        """
        with open(input_file, 'r', encoding='utf-8') as f: #read from input file
            lines = f.readlines()

        results = {}
        for idx, line in enumerate(lines):
            sentence = line.strip() #remove leading/trailing whitespaces or newline character
            tokens = self.tokenizer.tokenize(sentence)  #tokenize the sentence with tokenizer
            results[idx] = tokens  #store tokenized sentence with index as the key (starting from 0)

        with open(output_file, 'w', encoding='utf-8') as f: #save in output file
            json.dump(results, f, indent=2)

    def preprocess_data(self, corpus_path: str, vocab_size: int) -> List[Tuple[List[np.ndarray], int]]:
        """
        Preprocesses the data for training neural language models. Constructs vocabulary from corpus
        of vocab_size and then tokenize the corpus. Creates word (token) to index mapping and index to 
        word(token) mapping for vocabulary. Creates data samples in form of (context_embed, target_idx),
        where context_embed is the embeddings of the previous context_size tokens provided by Word2VecModel 
        and target_idx is the index of the word that should come after the context words.
        """
        # self.tokenizer.construct_vocabulary(corpus_path, vocab_size=vocab_size)
        # self.tokenize_txt_file(corpus_path, "task3-files/tokenized_corpus.json")

        with open("task2-files/tokenized_corpus.json", 'r', encoding='utf-8') as f: #update before submission
            tokenized_corpus = json.load(f)
            self.tokenized_sentences = [tokens for tokens in tokenized_corpus.values()] #list of tokenized sentences

        self.word2idx = {word: idx for idx, word in enumerate(self.tokenizer.vocab)} #token to index mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()} #index to token mapping
        data = []
        pad_idx = self.word2idx[self.pad_token] #index of pad_token in the vocabulary

        #creating (context_embed, target_idx) samples for each tokenized sentence
        #considering each token as target once, using pad token in case not enough tokens are available for context
        for sentence in self.tokenized_sentences:
            for i in range(len(sentence)):
                if i < self.context_size: #not enough context tokens
                    #pad the context index array with pad token index
                    context_idx = [pad_idx] * (self.context_size - i) + [self.word2idx[token] for token in sentence[:i]]
                else: #enough context tokens
                    context_idx = [self.word2idx[token] for token in sentence[i-self.context_size:i]]
                #get embeddings for context tokens
                context_embed = [self.word2vec.embeddings(torch.tensor(idx, dtype=torch.long)).detach().numpy() for idx in context_idx]
                target_idx = self.word2idx[sentence[i]] #target token index
                data.append((context_embed, target_idx))
        return data

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the context and target at the given index.
        """
        context, target = self.data[idx]
        context_tensor = torch.tensor(np.concatenate(context).flatten(), dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return context_tensor, target_tensor


class NeuralLM1(nn.Module):
    """
    Implements a simple neural language model with 1 hidden layer
    and a Tanh activation function for next-word prediction task.
    """
    def __init__(self, input_dim, hidden_dim, vocab_size):
        """
        Initilizes the NeuralLM1 model.
        """
        super(NeuralLM1, self).__init__()
        self.architecture = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        return self.architecture(x)


class NeuralLM2(nn.Module):
    """
    Implements a neural language model with 1 hidden layer and a Tanh activation
    function alongisde skip-connection for next-word prediction task.
    """
    def __init__(self, input_dim, hidden_dim, vocab_size):
        """
        Initilizes the NeuralLM2 model.
        """
        super(NeuralLM2, self).__init__()
        self.architecture = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        self.projection = nn.Linear(input_dim, vocab_size)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        identity = self.projection(x)
        out = self.architecture(x)
        out += identity
        return out


class NeuralLM3(nn.Module):
    """
    Implements a complex neural language model with 3 hidden layer and ReLU activations
    function alongisde normalization and dropout for next-word prediction task.
    """
    def __init__(self, input_dim, hidden_dim, vocab_size):
        """
        Initilizes the NeuralLM3 model.
        """
        super(NeuralLM3, self).__init__()
        self.architecture = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.BatchNorm1d(hidden_dim[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.BatchNorm1d(hidden_dim[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.BatchNorm1d(hidden_dim[2]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim[2], vocab_size)
        )
    
    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        return self.architecture(x)


def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes the accuracy of the model given the predictions and true labels.
    """
    _, preds = torch.max(predictions, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)

def compute_perplexity(loss: float) -> np.float64:
    """
    Computes the perplexity given the loss.
    """
    return np.exp(loss)


def train(model: NeuralLM1 | NeuralLM2 | NeuralLM3, train_dataloader: DataLoader, val_dataloader:DataLoader, epochs: int,
        lr: float, device: str) -> Tuple[List[float], List[float], List[np.float64], List[float], List[float], List[np.float64]]:
    """
    Function to train the neural language models.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #lists to store losses, accuracies and preplexity for each epoch
    train_losses, train_accuracies, train_preplexity = [], [], []
    val_losses, val_accuracies, val_preplexity = [], [], []

    for epoch in range(epochs):
        #training phase
        model.train()
        train_loss, train_acc = 0, 0
        for context, target in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            context, target = context.to(device), target.to(device)
            optimizer.zero_grad() #zero the gradients
            #forward pass
            outputs = model(context)
            loss = criterion(outputs, target)
            train_loss += loss.item()
            #backward pass and optimize
            loss.backward()
            optimizer.step()
            #compute batch accuracy
            train_acc += compute_accuracy(outputs, target)
        #calculate average training loss, accuracy and preplexity
        train_losses.append(train_loss / len(train_dataloader))
        train_accuracies.append(train_acc / len(train_dataloader))
        train_preplexity.append(compute_perplexity(train_losses[-1]))

        #validation phase
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for context, target in val_dataloader:
                context, target = context.to(device), target.to(device)
                #forward pass
                outputs = model(context)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                #compute batch accuracy
                val_acc += compute_accuracy(outputs, target)
        #calculate average validation loss, accuracy and preplexity
        val_losses.append(val_loss / len(val_dataloader))
        val_accuracies.append(val_acc / len(val_dataloader))
        val_preplexity.append(compute_perplexity(val_losses[-1]))

        #print losses for each epoch
        # print(f"----- Epoch {epoch + 1}/{epochs} -----")
        print(f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {100*train_accuracies[-1]:.2f}%, Train Perplexity: {train_preplexity[-1]:.2f}")
        print(f"Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {100*val_accuracies[-1]:.2f}%, Val Perplexity: {val_preplexity[-1]:.2f}")
        print()

    return train_losses, train_accuracies, train_preplexity, val_losses, val_accuracies, val_preplexity

def plot_losses(train_loss: List[float], val_loss: List[float], model: str, save_path: str) -> None:
    """
    Function to plot the training and validation losses
    """
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(model + ' - Epoch vs Loss')
    plt.legend()
    plt.savefig(save_path+model+'Loss.png')
    plt.close()


def predict_tokens(sentence: str, num_tokens: int, context_size: int, dataset: NeuralLMDataset,
                   model: NeuralLM1 | NeuralLM2 | NeuralLM3, device: str) -> List[str]:
    """
    Function to predict next tokens given a sentence using the trained model.
    """
    model.to(device)
    model.eval() #evaluation mode for predicing tokens

    tokens = dataset.tokenizer.tokenize(sentence) #tokenize the input sentence
    if len(tokens) < context_size: #add pad tokens if not enough tokens for context
        tokens = [dataset.pad_token] * (context_size - len(tokens)) + tokens

    #get embeddings for tokenized sentence
    sentence_embeds = [dataset.word2vec.embeddings(torch.tensor(dataset.word2idx[token], dtype=torch.long)).detach().numpy() for token in tokens]

    predicted_tokens = []
    for _ in range(num_tokens):
        context = sentence_embeds[-context_size:] #select previous context_size tokens for prediction
        context_tensor = torch.tensor(np.concatenate(context).flatten(), dtype=torch.float32)
        output = model(context_tensor)
        predicted_idx = torch.argmax(output).item()
        predicted_token = dataset.idx2word[predicted_idx]
        #append the new token and its embedding to the lists
        predicted_tokens.append(predicted_token)
        sentence_embeds.append(dataset.word2vec.embeddings(torch.tensor(predicted_idx, dtype=torch.long)).detach().numpy())

    return predicted_tokens

def prediciton_pipeline(input_file: str, num_tokens: int, context_size: int, dataset: NeuralLMDataset,
                        model: NeuralLM1 | NeuralLM2 | NeuralLM3, device: str) -> None:
    """
    Implements the pipeline to make next-word predictions for sentences in a text file.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    for sentence in sentences:
        sentence = sentence.strip() #remove newline character and whitespaces
        predicted_tokens = predict_tokens(sentence=sentence, num_tokens=num_tokens, context_size=context_size,
                                          dataset=dataset, model=model, device=device)
        print(f"\nInput: {sentence}")
        print(f"Predicted Tokens: {' '.join(predicted_tokens)}")


def load_model(model_class):
    model_path = "task2-files/final_model/final_model.pt"
    checkpoint = torch.load(model_path)
    model = model_class(vocab_size=checkpoint['vocab_size'], embedding_dim=checkpoint['embedding_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    val_loss = checkpoint['val_loss']
    val_accuracy = checkpoint['val_accuracy']
    return model, val_loss, val_accuracy

if __name__ == '__main__':
    #configuration variables
    CORPUS_PATH = "corpus.txt"
    PAD_TOKEN = '[PAD]'
    VOCAB_SIZE = 8500
    CONTEXT_SIZE = 4
    NUM_EPOCHS = 1
    LEARNING_RATE = 0.02
    BATCH_SIZE = 1024
    TRAIN_SPLIT = 0.8
    PLOT_SAVE_PATH = "task3-files/"
    PREDICT_NUM_TOKENS = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #create tokenizer object and load trained word2vec model
    # Tokenizer = WordPieceTokenizer()
    Tokenizer = pickle.load(open("task1-files/tokenizer.pkl", 'rb')) #update before submission
    Word2Vec, val_loss, val_accuracy = load_model(Word2VecModel)

    #create dataset and split into training and valdiaiton
    dataset = NeuralLMDataset(corpus_path=CORPUS_PATH, pad_token=PAD_TOKEN, vocab_size=VOCAB_SIZE,
                              tokenizer=Tokenizer, word2vec=Word2Vec, context_size=CONTEXT_SIZE)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[TRAIN_SPLIT, 1-TRAIN_SPLIT],
                                                               generator=torch.Generator().manual_seed(42))

    #create dataloaders for training set and validation set
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #create model instances and train
    # model1 = NeuralLM1(input_dim=Word2Vec.embedding_dim*CONTEXT_SIZE, hidden_dim=1024, vocab_size=VOCAB_SIZE)
    # train_losses_1, train_accuracies_1, train_preplexity_1, val_losses_1, val_accuracies_1, val_preplexity_1 = train(
    #     model=model1, train_dataloader=train_loader, val_dataloader=val_loader, epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=DEVICE)

    model2 = NeuralLM2(input_dim=Word2Vec.embedding_dim*CONTEXT_SIZE, hidden_dim=1024, vocab_size=VOCAB_SIZE)
    train_losses_2, train_accuracies_2, train_preplexity_2, val_losses_2, val_accuracies_2, val_preplexity_2 = train(
        model=model2, train_dataloader=train_loader, val_dataloader=val_loader, epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=DEVICE)

    # model3 = NeuralLM3(input_dim=Word2Vec.embedding_dim*CONTEXT_SIZE, hidden_dim=[256,1024,4096], vocab_size=VOCAB_SIZE)
    # train_losses_3, train_accuracies_3, train_preplexity_3, val_losses_3, val_accuracies_3, val_preplexity_3 = train(
    #     model=model3, train_dataloader=train_loader, val_dataloader=val_loader, epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=DEVICE)

    #plot losses
    # plot_losses(train_loss=train_losses_1, val_loss=val_losses_1, model='NeuralLM1', save_path=PLOT_SAVE_PATH)
    # plot_losses(train_loss=train_losses_2, val_loss=val_losses_2, model='NeuralLM2', save_path=PLOT_SAVE_PATH)
    # plot_losses(train_loss=train_losses_3, val_loss=val_losses_3, model='NeuralLM3', save_path=PLOT_SAVE_PATH)

    #use prediction pipeline to predict next tokens for each sentence in a text file
    # prediciton_pipeline(input_file='task3-files/sample_test.txt', num_tokens=PREDICT_NUM_TOKENS,
    #                     context_size=CONTEXT_SIZE, dataset=dataset, model=model1, device=DEVICE)