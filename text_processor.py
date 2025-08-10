"""
Text Processor Module for AI Paper Keyword Extractor
Handles text cleaning, preprocessing, and preparation for keyword extraction
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from typing import List
import string


class TextProcessor:
    """Class to handle text preprocessing and cleaning"""
    
    def __init__(self):
        """Initialize the text processor and download required NLTK data"""
        self.lemmatizer = WordNetLemmatizer()
        self._download_nltk_resources()
        
        # Try to get stopwords, fallback to basic set if not available
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Warning: NLTK stopwords not available, using basic fallback")
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 
                'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                'further', 'then', 'once'
            }
        
        # Academic paper specific stop words
        self.academic_stop_words = {
            'abstract', 'introduction', 'conclusion', 'results', 'discussion', 
            'methodology', 'method', 'methods', 'figure', 'table', 'section',
            'paper', 'study', 'research', 'analysis', 'approach', 'work',
            'authors', 'article', 'journal', 'conference', 'proceedings',
            'university', 'department', 'email', 'corresponding', 'author',
            'pp', 'vol', 'no', 'doi', 'isbn', 'issn', 'et', 'al', 'etc',
            'i.e', 'e.g', 'cf', 'vs', 'fig', 'ref', 'refs', 'equation',
            'formula', 'algorithm', 'step', 'steps', 'first', 'second', 'third',
            'finally', 'moreover', 'however', 'therefore', 'thus', 'hence'
        }
        
        # Combine standard and academic stop words
        self.all_stop_words = self.stop_words.union(self.academic_stop_words)
    
    def _download_nltk_resources(self):
        """Download required NLTK resources"""
        import ssl
        
        # Handle SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        required_resources = [
            'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4'
        ]
        
        for resource in required_resources:
            try:
                # Try to find the resource first
                if resource == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                elif resource == 'stopwords':
                    nltk.data.find('corpora/stopwords')
                elif resource == 'wordnet':
                    nltk.data.find('corpora/wordnet')
                elif resource == 'averaged_perceptron_tagger':
                    nltk.data.find('taggers/averaged_perceptron_tagger')
                elif resource == 'omw-1.4':
                    nltk.data.find('corpora/omw-1.4')
            except LookupError:
                try:
                    print(f"Downloading NLTK resource: {resource}")
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    print(f"Warning: Could not download {resource}: {e}")
                    print("Some features may not work properly.")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Remove citations in common formats [1], (Smith, 2020), etc.
        text = re.sub(r'\[[0-9,\s-]+\]', '', text)
        text = re.sub(r'\([A-Za-z\s,]+\s\d{4}[a-z]?\)', '', text)
        
        # Remove figure and table references
        text = re.sub(r'[Ff]igure\s+\d+[a-z]?', '', text)
        text = re.sub(r'[Tt]able\s+\d+[a-z]?', '', text)
        text = re.sub(r'[Ss]ection\s+\d+(\.\d+)*', '', text)
        
        # Remove mathematical expressions and equations
        text = re.sub(r'\$[^$]*\$', '', text)  # LaTeX math
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)  # LaTeX commands
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?]', ' ', text)
        
        # Remove standalone numbers and short fragments
        text = re.sub(r'\b\d+\b', '', text)
        text = re.sub(r'\b[a-zA-Z]\b', '', text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_filter(self, text: str, min_length: int = 3) -> List[str]:
        """
        Tokenize text and filter out stop words and short words
        
        Args:
            text (str): Text to tokenize
            min_length (int): Minimum word length to keep
            
        Returns:
            List[str]: Filtered tokens
        """
        # Tokenize - fallback to simple split if NLTK tokenizer fails
        try:
            tokens = word_tokenize(text.lower())
        except LookupError:
            # Simple fallback tokenization
            tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            if (len(token) >= min_length and 
                token not in self.all_stop_words and 
                token not in string.punctuation and
                token.isalpha()):  # Only alphabetic tokens
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from text using POS tagging
        
        Args:
            text (str): Text to process
            
        Returns:
            List[str]: List of noun phrases
        """
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            # Simple fallback sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        noun_phrases = []
        
        for sentence in sentences:
            try:
                tokens = word_tokenize(sentence.lower())
                pos_tags = pos_tag(tokens)
                
                current_phrase = []
                for word, tag in pos_tags:
                    if tag.startswith('NN') or tag.startswith('JJ'):  # Nouns and adjectives
                        if (word not in self.all_stop_words and 
                            len(word) > 2 and 
                            word.isalpha()):
                            current_phrase.append(word)
                    else:
                        if len(current_phrase) >= 2:  # Multi-word phrases
                            noun_phrases.append(' '.join(current_phrase))
                        current_phrase = []
                
                # Add final phrase if exists
                if len(current_phrase) >= 2:
                    noun_phrases.append(' '.join(current_phrase))
                    
            except LookupError:
                # Simple fallback - find consecutive non-stop words
                words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
                current_phrase = []
                for word in words:
                    if (word not in self.all_stop_words and 
                        len(word) > 2 and 
                        word.isalpha()):
                        current_phrase.append(word)
                    else:
                        if len(current_phrase) >= 2:
                            noun_phrases.append(' '.join(current_phrase))
                        current_phrase = []
                
                if len(current_phrase) >= 2:
                    noun_phrases.append(' '.join(current_phrase))
        
        return noun_phrases
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base forms
        
        Args:
            tokens (List[str]): List of tokens to lemmatize
            
        Returns:
            List[str]: Lemmatized tokens
        """
        lemmatized = []
        for token in tokens:
            lemmatized_token = self.lemmatizer.lemmatize(token)
            lemmatized.append(lemmatized_token)
        
        return lemmatized
    
    def preprocess_for_keywords(self, text: str) -> dict:
        """
        Complete preprocessing pipeline for keyword extraction
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            dict: Processed text components
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Get tokens
        tokens = self.tokenize_and_filter(cleaned_text)
        
        # Lemmatize tokens
        lemmatized_tokens = self.lemmatize_tokens(tokens)
        
        # Extract noun phrases
        noun_phrases = self.extract_noun_phrases(cleaned_text)
        
        # Reconstruct cleaned text for TF-IDF
        processed_text = ' '.join(lemmatized_tokens)
        
        return {
            'cleaned_text': cleaned_text,
            'tokens': lemmatized_tokens,
            'noun_phrases': noun_phrases,
            'processed_text': processed_text,
            'word_count': len(lemmatized_tokens),
            'unique_words': len(set(lemmatized_tokens))
        }


if __name__ == "__main__":
    # Test the text processor
    processor = TextProcessor()
    
    sample_text = """
    Abstract: This paper presents a novel approach to machine learning algorithms 
    for natural language processing. We propose a new method that significantly 
    improves performance over existing approaches. The experimental results show 
    that our algorithm achieves state-of-the-art performance on multiple datasets.
    
    1. Introduction
    Machine learning has revolutionized the field of artificial intelligence.
    Recent advances in deep learning have shown promising results.
    """
    
    print("Processing sample text...")
    result = processor.preprocess_for_keywords(sample_text)
    
    print(f"Word count: {result['word_count']}")
    print(f"Unique words: {result['unique_words']}")
    print(f"Top tokens: {result['tokens'][:10]}")
    print(f"Noun phrases: {result['noun_phrases'][:5]}")
