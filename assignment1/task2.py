import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import json
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import os

from task1 import WordPieceTokenizer

torch.manual_seed(42)
torch.cuda.manual_seed(42)


'''Word2VecDataset Class'''
class Word2VecDataset(Dataset):
    
    def __init__(self, window_size, vocabulary_size):

        self.pad_token = '[PAD]'
        self.window_size = window_size
        self.tokenizer = WordPieceTokenizer()
        self.vocabulary_size = vocabulary_size

        self.text = None

        # stores unique words in the corpus
        self.vocabulary = set()
        # a dictionary mapping words to indices
        self.word_to_idx = {}
        # a dictionary mapping indices to words
        self.idx_to_word = {}
        # a list that will store training pairs
        self.cbow_pairs = []
        
        self.preprocess_data()

    '''uses Word Piece Tokenizer from Task 1 to tokenize input corpus
    def tokenize_txt_file(self, input_file, output_file):
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        results = {}
        
        for idx, line in enumerate(lines):
            sentence = line.strip() 
            tokens = self.tokenizer.tokenize(sentence)
            results[str(idx)] = tokens 

        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
    '''

    
    '''updates vocabulary'''
    def update_vocabulary(self):

        self.tokenizer.construct_vocabulary("corpus.txt", vocab_size=self.vocabulary_size)

        vocabulary = []

        with open('task1-files/vocabulary_35.txt', 'r') as file:
            
            for line in file:
                word = line.strip()
                if word:  # to avoid adding empty lines
                    vocabulary.append(word)

        self.vocabulary = vocabulary
        print("got and updated vocab")

    
    def tokenize_corpus(self):
        
        # get the tokenized corpus file using WordPieceTokenizer from task 1
        input = "corpus.txt"
        output = "task2-files/tokenized_corpus.json"

        with open(input, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        results = {}
        
        for idx, line in enumerate(lines):
            sentence = line.strip() 
            tokens = self.tokenizer.tokenize(sentence)
            results[str(idx)] = tokens 

        
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        # converting json tokenized corpus file to 2d array and updating dataset corpus
        corpus = None

        with open('task2-files/tokenized_corpus.json', 'r', encoding='utf-8') as f:

            tokenized_corpus = json.load(f)
            # convert to list 2d
            corpus = [tokens for tokens in tokenized_corpus.values()]
            
        
        self.text = corpus

    
    '''function to generate cbow_pairs'''
    def generate_cbow_pairs(self):

        # looping over tokenized corpus
        for line in self.text:

            for j in range(len(line)):

                #context words
                start = max(0, j-self.window_size)
                end = min(len(line), j + self.window_size + 1)

                context_words = (line[start:j] + line[j + 1:end])
                
                
                if len(context_words) > 0:

                    # pad if needed
                    while len(context_words) < self.window_size * 2:
                        context_words.append(self.pad_token)
                    
                    # get indices for context words
                    context_idxs = []

                    for word in context_words:

                        word_index = self.word_to_idx.get(word, self.word_to_idx[self.pad_token])
                        context_idxs.append(word_index)

                    # get index for target word
                    target_idx = self.word_to_idx.get(line[j], self.word_to_idx[self.pad_token])
                    
                    # store pair
                    self.cbow_pairs.append((context_idxs, target_idx))


    '''function to preprocess the data using above defined functions'''
    def preprocess_data(self):

        print("updating vocabulary")     
        self.update_vocabulary()

        print("tokenizing corpus")
        self.tokenize_corpus()

        print("updating mapping - word2idx and idx2word")

        # updates the word to index mapping
        # self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}

        # Create a dictionary that maps each word in the vocabulary to a unique index
        self.word_to_idx = {}

        for idx, word in enumerate(self.vocabulary):
            self.word_to_idx[word] = idx

        
        # updates the reverse index to word mapping
        self.idx_to_word = {}

        for word, idx in self.word_to_idx.items():
            self.idx_to_word[idx] = word
        

        print("generating cbow pairs")
        self.generate_cbow_pairs()
    
    # returns the total number of CBOW training pairs
    def __len__(self):
        return len(self.cbow_pairs)
    
    
    # used to retrieve a single CBOW pair (context, target) pair
    def __getitem__(self, idx):

        # get the pair of ith index as mentioned parameter 
        context_indices, target_idx = self.cbow_pairs[idx]
        
        # Convert to tensors with explicit types
        context_tensor = torch.tensor(context_indices, dtype=torch.long)
        target_tensor = torch.tensor(target_idx, dtype=torch.long)
        
        return context_tensor, target_tensor



'''Word2VecModel'''
class Word2VecModel(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim):

        super().__init__()
        
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        
        # table that stores word representations for each word
        self.embeddings = nn.Embedding(vocabulary_size, embedding_dim)

        self.output_layer = nn.Linear(embedding_dim, vocabulary_size)
        

        self._initialize_weights()
    
    '''function to initialize weight'''
    def _initialize_weights(self):
        
        initrange = 0.5 / self.embedding_dim
        
        # initializing embedding layer
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        
        # initializing output layer
        self.output_layer.weight.data.uniform_(-initrange, initrange)

        # initializing the bias of output layer
        self.output_layer.bias.data.zero_()
    

    # function for forward pass of the model
    def forward(self, x: torch.Tensor):

        """
        x : input tensor, which represents the indices of the context words in the vocabulary. Its shape is [batch size, window size*2] -- [target word, [list of indexes of context words]]
        """

        # embeddings for each word in context and output list of embeddings of each context word
        # size = [batch size, window size*2, embedding dimension]
        embedded = self.embeddings(x)
        
        # calculating average of context words... horizontally - row by row
        # size = [batch_size, embedding_dim]
        context_embedding = torch.mean(embedded, dim=1)
        
        # passing the average context embedding to output layer
        # output size vector = [batch_size, vocabulary_size]
        output = self.output_layer(context_embedding)

        # returning after application of softmax function
        return F.log_softmax(output, dim=1)

    
    '''get embedding vector for a word'''
    def get_word_vector(self, word_idx):

        return self.embeddings.weight.data[word_idx]
    

    '''function to get cosine similarity of between two vectors'''
    def cosine_similarity(self, word_idx1, word_idx2):

        vec1 = self.get_word_vector(word_idx1)
        vec2 = self.get_word_vector(word_idx2)
        
        return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
    

    '''function to get the two most similar vectors to the given vector'''
    def find_most_similar(self, word_idx):

        target_vector = self.get_word_vector(word_idx)
        
        # calculating similarities with all words
        similarities = F.cosine_similarity(
            target_vector.unsqueeze(0),
            self.embeddings.weight.data
        )
        
        # excluding the word itself
        similarities[word_idx] = -1

        # get the top two similar vectors
        top_scores, top_indices = similarities.topk(2)
        
        return list(zip(top_indices.tolist(), top_scores.tolist()))
    
    
    '''function to get the least similar vector to the given vector'''
    def find_least_similar(self, word_idx):
 
        target_vector = self.get_word_vector(word_idx)
        
        # calculating similarities with all words
        similarities = F.cosine_similarity(
            target_vector.unsqueeze(0),
            self.embeddings.weight.data
        )
        
        # excluding the word itself
        similarities[word_idx] = 1
        
        # get the least similar vector
        min_score, min_idx = similarities.min(dim=0)
        
        return (min_idx.item(), min_score.item())



'''Function to validate model after every accuracy
returns - validation loss, validation accuracy and average cosine similarity'''

def get_avg_cosine_similarity(cos_sim, predicted, target): 
    total_cosine_sim = 0.0
    # Calculate cosine similarity
    # Get embeddings for predicted and target words
    predicted_embeddings = model.embeddings(predicted)
    target_embeddings = model.embeddings(target)
            
    # Calculate cosine similarity between predicted and target embeddings
    batch_cosine_sim = cos_sim(predicted_embeddings, target_embeddings).mean()
    total_cosine_sim += batch_cosine_sim.item()

    return total_cosine_sim

def validate_model(model, val_loader, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    
    model.eval()
    criterion = nn.NLLLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_cosine_sim = 0.0
    num_batches = len(val_loader)
    
    # Cosine similarity function
    cos_sim = nn.CosineSimilarity(dim=1)
    
    with torch.no_grad():
        for context, target in val_loader:
            context = context.to(device)
            target = target.to(device)
            
            # Forward pass
            output = model(context)
            loss = criterion(output, target)
            
            # Calculate loss
            total_loss += loss.item()
            
            # Calculate accuracy
            predicted = output.argmax(dim=1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)
            
            '''# Calculate cosine similarity
            # Get embeddings for predicted and target words
            predicted_embeddings = model.embeddings(predicted)
            target_embeddings = model.embeddings(target)
            
            # Calculate cosine similarity between predicted and target embeddings
            batch_cosine_sim = cos_sim(predicted_embeddings, target_embeddings).mean()
            total_cosine_sim += batch_cosine_sim.item()'''

            total_cosine_sim += get_avg_cosine_similarity(cos_sim, predicted, target)
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    accuracy = (total_correct / total_samples) * 100
    avg_cosine_sim = total_cosine_sim / num_batches
    
    return avg_loss, accuracy, avg_cosine_sim



'''def evaluate_model(model, val_loader, device, dataset, BATCH_SIZE) :
    model.eval()
    
    with torch.no_grad():
        for i, (context, target) in enumerate(val_loader):
            context = context.to(device)
            target = target.to(device)
            
            # Get model prediction
            output = model(context)
            predicted_indices = output.argmax(dim=1)
            
            # Convert to words
            for j in range(len(context)):
                context_words = [dataset.idx_to_word[idx.item()] for idx in context[j]]
                true_word = dataset.idx_to_word[target[j].item()]
                predicted_word = dataset.idx_to_word[predicted_indices[j].item()]
                
                print(f"\nPair {i*BATCH_SIZE + j + 1}:")
                print(f"Context: {context_words}")
                print(f"True word: {true_word}")
                print(f"Predicted: {predicted_word}")
                print(f"Correct: {'✓' if true_word == predicted_word else '✗'}")
            
            # Print only first 5 validation pairs for brevity
            if i >= 4:
                print("\n... (showing first 20 pairs only)")
                break
'''

def save_checkpoint(model, checkpoint_path, epoch, optimizer, loss, accuracy):

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'vocabulary_size': model.vocabulary_size,
        'embedding_dim': model.embedding_dim,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, checkpoint_path, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return {
        'model': model,
        'optimizer_state_dict': checkpoint['optimizer_state_dict'],
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'], 
        'accuracy': checkpoint['accuracy'], 
        'vocabulary_size': checkpoint['vocabulary_size'],
        'embedding_dim': checkpoint['embedding_dim'],
        'timestamp': checkpoint['timestamp']
    }




'''training function'''
def train_word2vec(model,train_loader,val_loader,num_epochs,learning_rate,
    checkpoint_dir = 'checkpoints',
    save_frequency = 5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    history = {
        'epoch_losses': [],
        'epoch_accuracies': [],
        'batch_losses': [],
        'batch_accuracies': [],
        'val_losses':[],
        'val_accuracies':[],
        'val_similarities':[]
    }
    
    for epoch in range(num_epochs):
        # Compute decayed learning rate
        current_lr = learning_rate * (1 - epoch / num_epochs)
   
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # initializing these for tracking
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_batches = len(train_loader)
        
        # displaying progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        # setting the model to training mode
        model.train()

        for batch_idx, (context, target) in enumerate(progress_bar):

            # moving data to the appropriate device
            context, target = context.to(device), target.to(device)
            optimizer.zero_grad()

            # forward pass
            output = model.forward(context)

            # calculating loss
            loss = criterion(output, target)

            # backward pass
            loss.backward()
            optimizer.step()
            
            # updating batch accuracy and loss
            predicted = output.argmax(dim=1)
            correct = (predicted == target).sum().item()
            total = target.size(0)
            
            batch_accuracy = (correct / total) * 100
            batch_loss = loss.item()

            # updating tracking vars
            epoch_loss += batch_loss
            epoch_correct += correct
            epoch_total += total
            
            history['batch_losses'].append(batch_loss)
            history['batch_accuracies'].append(batch_accuracy)
            
            progress_bar.set_postfix({
                'batch_loss': f'{batch_loss:.4f}',
                'batch_acc': f'{batch_accuracy:.2f}%',
                'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}',
                'lr': f'{current_lr:.6f}'
            })
        
        # storing and printing training loss
        avg_epoch_loss = epoch_loss / num_batches
        epoch_accuracy = (epoch_correct / epoch_total) * 100

        history['epoch_losses'].append(avg_epoch_loss)
        history['epoch_accuracies'].append(epoch_accuracy)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Average Loss: {avg_epoch_loss:.4f}')
        print(f'Training Accuracy: {epoch_accuracy:.2f}%')
        
        # storing and printing validation loss
        print("\nValidating model...")
        val_loss, accuracy, cosine_similarity = validate_model(model, val_loader, device)

        history['val_losses'].append(val_loss)
        history['val_accuracies'].append(val_loss)
        history['val_similarities'].append(val_loss)

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.2f}%")
        print(f"Cosine Similarity: {cosine_similarity}")
        
        
        '''# saving checkpoints
        if (epoch + 1) % save_frequency == 0:

            checkpoint_path = os.path.join(checkpoint_dir, f'word2vec_checkpoint_epoch_{epoch+1}.pt')

            save_checkpoint(model, checkpoint_path, epoch, optimizer, avg_epoch_loss, epoch_accuracy)

            print(f'Checkpoint saved: {checkpoint_path}')'''
    
    # saving final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, 'word2vec_final_model.pt')
    save_checkpoint(model, final_checkpoint_path, num_epochs-1, optimizer, history['epoch_losses'][-1], history['epoch_accuracies'][-1])
    
    print(f'Final model saved: {final_checkpoint_path}')
    
    return history




def plot_training_history(history):
    
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history['epoch_losses'], label='Training Loss')
    ax1.set_title('Training Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax3.plot(history['val_losses'], label='Validation Loss')
    ax3.set_title('Validation Loss Over Epochs')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()


'''function to save final model'''
def save_model(final_model_dir, model, val_loss, accuracy):
    
    model_path = os.path.join(final_model_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocabulary_size': model.vocabulary_size,
        'embedding_dim': model.embedding_dim,
        'val_loss': val_loss,
        'val_accuracy': accuracy
    }, model_path)


'''function to save generated vocabulary'''
def save_vocabulary(final_model_dir, dataset):  

    vocab_path = os.path.join(final_model_dir, 'vocabulary.json')
    vocab_data = {
        'word2idx': dataset.word_to_idx,
        'idx2word': dataset.idx_to_word
    }
    with open(vocab_path, 'w') as f:
        json.dump(vocab_data, f)



'''function to get triplets'''
def find_triplets(model: Word2VecModel, num_triplets: int = 2):
    triplets = []
    
    for i in range(num_triplets):
        # randomly choose an initial word index
        word_idx = torch.randint(0, model.vocabulary_size, (1,)).item()
        
        # get most similar words
        most_similar = model.find_most_similar(word_idx)
        
        # get the least similar word
        least_similar = model.find_least_similar(word_idx)
        
        # construct the triplet
        triplet = (word_idx, most_similar[0][0], most_similar[1][0], least_similar[0])
        triplets.append(triplet)
    
    return triplets

if __name__ == "__main__":

    WINDOW_SIZE = 4
    EMBEDDING_DIM = 10
    BATCH_SIZE = 256
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.02
    TRAIN_SPLIT = 0.8    
    VOCABULARY_SIZE = 8500

    # Create dataset
    print("Creating dataset...")
    dataset = Word2VecDataset(window_size=WINDOW_SIZE, vocabulary_size=VOCABULARY_SIZE)
        

    # Split dataset into training and validation
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        

    # Print dataset information
    print(f"\nVocabulary Size: {len(dataset.vocabulary)}")
    print(f"Total Pairs: {len(dataset)}")
    print(f"Training Pairs: {len(train_dataset)}")
    print(f"Validation Pairs: {len(val_dataset)}")
    print("\nSample vocabulary items:", list(dataset.vocabulary)[:5])
        

    # model initialization
    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Word2VecModel(vocabulary_size=len(dataset.vocabulary), embedding_dim=EMBEDDING_DIM).to(device)
        

    # training model
    print("\nStarting training...")

    history = train_word2vec(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            checkpoint_dir='task2-files/word2vec_checkpoints',
            save_frequency=2,
            device=device
    )
        

    # Validate model and get accuracy
    print("\nValidating model...")
    val_loss, accuracy, avg_cosine = validate_model(model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.2f}%")
        

    # Plot training history
    plot_training_history(history)
        

    ''' # Print validation pairs and predictions
    print("\nValidation Pairs vs Predictions:")
    print("-" * 50)

    evaluate_model(model, val_loader, device, dataset, BATCH_SIZE)'''
        

    # Save final model
    final_model_dir = 'task2-files/final_model'
    os.makedirs(final_model_dir, exist_ok=True)


    # save vocabulary  
    save_vocabulary(final_model_dir, dataset)
    # save model
    save_model(final_model_dir, model, val_loss, accuracy)



    # getting triplets
    triplets = find_triplets(model)

    print("Identified Triplets:")

    for triplet in triplets:
        print(f"\nToken - {triplet[0]}\nSimilar tokens - {triplet[1]}, {triplet[2]}\nDissimilar token - {triplet[3]}")

        sim1 = model.cosine_similarity(triplet[0], triplet[1])
        sim2 = model.cosine_similarity(triplet[0], triplet[2])
        sim3 = model.cosine_similarity(triplet[0], triplet[3])

        print(f"Cosine similarity ({triplet[0]}, {triplet[1]}): {sim1}")
        print(f"Cosine similarity ({triplet[0]}, {triplet[2]}): {sim2}")
        print(f"Cosine similarity ({triplet[0]}, {triplet[3]}): {sim3}")