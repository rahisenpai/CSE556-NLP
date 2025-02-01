import json
from collections import defaultdict
import re
from typing import List, Dict, Tuple
import pickle


class WordPieceTokenizer:
    def __init__(self):
        self.vocab = {"[UNK]", "[PAD]"}
        self.split_words = []  # To store the current state of split words
        
    def preprocess_data(self, text: str) -> str:
        # Convert to lowercase (this corpus does not need this part of pre-processing)
        text = text.lower()
        
        # Add spaces around punctuation
        text = re.sub(r'([.,!?()])', r' \1 ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip() # Only this part of pre-processing is needed for this corpus
    
    def split_word(self, word: str) -> List[str]:
        """ Split word into characters, adding ## prefix to non-initial characters """
        # similar to BPE Algorithm
        if not word:
            return []
        chars = list(word)
        return [chars[0]] + [f"##{c}" for c in chars[1:]]
    
    def get_pair_frequencies(self, split_words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """ Count frequencies of adjacent pairs in the split words """
        pair_freqs = defaultdict(int)
        for word in split_words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] += 1
        return pair_freqs
    
    def get_unit_frequencies(self, split_words: List[List[str]]) -> Dict[str, int]:
        """ Count frequencies of individual units """
        unit_freqs = defaultdict(int)
        for word in split_words:
            for unit in word:
                unit_freqs[unit] += 1
        return unit_freqs
    
    def calculate_pair_scores(self, pair_freqs: Dict[Tuple[str, str], int], 
                             unit_freqs: Dict[str, int]) -> Dict[Tuple[str, str], float]:
        """ Calculate scores for each pair using the formula: freq(pair) / (freq(first) * freq(second)) """
        scores = {}
        for pair, freq in pair_freqs.items():
            first, second = pair
            score = freq / (unit_freqs[first] * unit_freqs[second])
            scores[pair] = score
        return scores
    
    def merge_pair(self, pair: Tuple[str, str], split_words: List[List[str]]) -> List[List[str]]:
        """ Merge the specified pair in all occurrences in the split words """
        first, second = pair
        merged = first + second.replace("##", "")
        
        new_split_words = []
        for word in split_words:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_split_words.append(new_word)
        
        return new_split_words
    
    def read_corpus(self, file_path: str) -> List[str]:
        """ Read corpus from text file """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()

    def construct_vocabulary(self, corpus_file: str, vocab_size: int) -> None:
        """
        Construct vocabulary using WordPiece algorithm using the specified scoring method
        """
        # Read and preprocess corpus
        corpus = self.read_corpus(corpus_file)
        processed_texts = [self.preprocess_data(text) for text in corpus]
        
        # Split all words into characters with ## prefix
        self.split_words = []
        for text in processed_texts:
            for word in text.split():
                self.split_words.append(self.split_word(word))
        
        # Add initial units to vocabulary
        initial_units = set()
        for word in self.split_words:
            initial_units.update(word)
        self.vocab.update(initial_units)
        
        # Main loop for merging pairs until vocabulary size is reached
        while len(self.vocab) < vocab_size:
            # Get frequencies
            pair_freqs = self.get_pair_frequencies(self.split_words)
            if not pair_freqs:
                break
                
            unit_freqs = self.get_unit_frequencies(self.split_words)
            
            # Calculate scores
            scores = self.calculate_pair_scores(pair_freqs, unit_freqs)
            if not scores:
                break
                
            # Find best pair
            best_pair = max(scores.items(), key=lambda x: x[1])[0]
            merged_token = best_pair[0] + best_pair[1].replace("##", "")
            
            # Add to vocabulary and merge in corpus
            self.vocab.add(merged_token)
            self.split_words = self.merge_pair(best_pair, self.split_words)
        
        # Save vocabulary to file
        with open('task1-files/vocabulary_35.txt', 'w', encoding='utf-8') as f:
            for token in sorted(self.vocab):
                f.write(f"{token}\n")
    
    def tokenize(self, text: str) -> List[str]:
        """ Tokenize input text using the constructed vocabulary """
        text = self.preprocess_data(text)
        result = []
        
        for word in text.split():
            # Character-level split
            current = self.split_word(word)
            
            # Merging using vocabulary as long as possible
            final_tokens = []
            i = 0
            while i < len(current):
                longest_match = None
                longest_length = 0
                
                # Try to find longest matching sequence starting at current position
                for j in range(i + 1, len(current) + 1):
                    candidate = current[i]
                    for k in range(i + 1, j):
                        candidate += current[k].replace("##", "")
                    if candidate in self.vocab and j - i > longest_length:
                        longest_match = candidate
                        longest_length = j - i
                
                if longest_match:
                    final_tokens.append(longest_match)
                    i += longest_length
                else:
                    # If no match found, mark as unknown
                    final_tokens.append("[UNK]")
                    i += 1
            
            result.extend(final_tokens)
        
        return result
    
    def tokenize_file(self, input_file: str, output_file: str) -> None:
        """Tokenize sentences from input JSON file and save results"""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = {}
        for item in data:
            id_ = item['id']
            sentence = item['sentence']
            tokens = self.tokenize(sentence)
            results[str(id_)] = tokens
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    tokenizer = WordPieceTokenizer()

    # Construct vocabulary from corpus.txt file
    tokenizer.construct_vocabulary("corpus.txt", vocab_size=8500)

    # Tokenize test file
    tokenizer.tokenize_file("task1-files/sample_test.json", "task1-files/tokenized_35.json")

    #save tokenizer object
    # with open('task1-files/tokenizer.pkl', 'wb') as f:
    #     pickle.dump(tokenizer, f)