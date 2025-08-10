"""
Keyword Extractor Module for AI Paper Keyword Extractor
Implements TF-IDF and RAKE algorithms for keyword extraction
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rake_nltk import Rake
from collections import Counter
from typing import List, Tuple, Dict
import re


class KeywordExtractor:
    """Class to extract keywords using multiple algorithms"""
    
    def __init__(self):
        """Initialize the keyword extractor"""
        self.tfidf_vectorizer = None
        self.rake = Rake(
            stopwords=['english'],
            min_length=2,
            max_length=4,
            include_repeated_phrases=False
        )
    
    def extract_tfidf_keywords(self, text: str, max_features: int = 100, 
                              top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF algorithm
        
        Args:
            text (str): Preprocessed text
            max_features (int): Maximum number of features for TF-IDF
            top_k (int): Number of top keywords to return
            
        Returns:
            List[Tuple[str, float]]: List of (keyword, score) tuples
        """
        if not text or not text.strip():
            return []
        
        try:
            # Split text into sentences for TF-IDF
            sentences = self._split_into_sentences(text)
            
            if len(sentences) < 2:
                sentences = [text]
            
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
                stop_words='english',
                lowercase=True,
                token_pattern=r'\b[a-zA-Z]{3,}\b'  # Only words with 3+ characters
            )
            
            # Fit and transform
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Calculate mean TF-IDF scores across all sentences
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Create keyword-score pairs
            keyword_scores = list(zip(feature_names, mean_scores))
            
            # Filter out low scores and sort
            keyword_scores = [(kw, score) for kw, score in keyword_scores if score > 0]
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return keyword_scores[:top_k]
            
        except Exception as e:
            print(f"Error in TF-IDF extraction: {e}")
            return []
    
    def extract_rake_keywords(self, text: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Extract keywords using RAKE algorithm
        
        Args:
            text (str): Raw or preprocessed text
            top_k (int): Number of top keywords to return
            
        Returns:
            List[Tuple[str, float]]: List of (keyword, score) tuples
        """
        if not text or not text.strip():
            return []
        
        try:
            # Extract keywords using RAKE
            self.rake.extract_keywords_from_text(text)
            keyword_scores = self.rake.get_ranked_phrases_with_scores()
            
            # Filter and clean keywords
            filtered_keywords = []
            for score, phrase in keyword_scores:
                # Clean the phrase
                phrase = phrase.strip().lower()
                
                # Filter criteria
                if (len(phrase) >= 3 and 
                    not phrase.isdigit() and
                    not self._is_stop_phrase(phrase)):
                    filtered_keywords.append((phrase, score))
            
            # Sort by score and return top k
            filtered_keywords.sort(key=lambda x: x[1], reverse=True)
            return filtered_keywords[:top_k]
            
        except Exception as e:
            print(f"Error in RAKE extraction: {e}")
            return []
    
    def extract_combined_keywords(self, text: str, processed_text: str, 
                                 tfidf_weight: float = 0.6, rake_weight: float = 0.4,
                                 top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Extract keywords using combined TF-IDF and RAKE approach
        
        Args:
            text (str): Raw text for RAKE
            processed_text (str): Preprocessed text for TF-IDF
            tfidf_weight (float): Weight for TF-IDF scores
            rake_weight (float): Weight for RAKE scores
            top_k (int): Number of top keywords to return
            
        Returns:
            List[Tuple[str, float]]: List of (keyword, combined_score) tuples
        """
        # Get keywords from both methods
        tfidf_keywords = self.extract_tfidf_keywords(processed_text, top_k=30)
        rake_keywords = self.extract_rake_keywords(text, top_k=30)
        
        # Normalize scores
        tfidf_normalized = self._normalize_scores(tfidf_keywords)
        rake_normalized = self._normalize_scores(rake_keywords)
        
        # Combine scores
        combined_scores = {}
        
        # Add TF-IDF scores
        for keyword, score in tfidf_normalized:
            combined_scores[keyword] = tfidf_weight * score
        
        # Add RAKE scores
        for keyword, score in rake_normalized:
            if keyword in combined_scores:
                combined_scores[keyword] += rake_weight * score
            else:
                combined_scores[keyword] = rake_weight * score
        
        # Sort and return top keywords
        sorted_keywords = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_k]
    
    def extract_noun_phrase_keywords(self, noun_phrases: List[str], 
                                   top_k: int = 10) -> List[Tuple[str, int]]:
        """
        Extract keywords from noun phrases based on frequency
        
        Args:
            noun_phrases (List[str]): List of noun phrases
            top_k (int): Number of top keywords to return
            
        Returns:
            List[Tuple[str, int]]: List of (phrase, frequency) tuples
        """
        if not noun_phrases:
            return []
        
        # Count phrase frequencies
        phrase_counter = Counter(noun_phrases)
        
        # Filter and sort
        filtered_phrases = []
        for phrase, count in phrase_counter.items():
            if (len(phrase.split()) >= 2 and  # Multi-word phrases
                count > 1 and  # Appears more than once
                len(phrase) >= 6):  # Minimum phrase length
                filtered_phrases.append((phrase, count))
        
        # Sort by frequency
        filtered_phrases.sort(key=lambda x: x[1], reverse=True)
        return filtered_phrases[:top_k]
    
    def get_keyword_summary(self, text: str, processed_data: dict, 
                          top_k: int = 15) -> dict:
        """
        Get comprehensive keyword summary using all methods
        
        Args:
            text (str): Raw text
            processed_data (dict): Preprocessed text data
            top_k (int): Number of top keywords for each method
            
        Returns:
            dict: Summary with keywords from different methods
        """
        summary = {}
        
        # TF-IDF keywords
        summary['tfidf_keywords'] = self.extract_tfidf_keywords(
            processed_data['processed_text'], top_k=top_k
        )
        
        # RAKE keywords
        summary['rake_keywords'] = self.extract_rake_keywords(text, top_k=top_k)
        
        # Combined keywords
        summary['combined_keywords'] = self.extract_combined_keywords(
            text, processed_data['processed_text'], top_k=top_k
        )
        
        # Noun phrase keywords
        summary['noun_phrase_keywords'] = self.extract_noun_phrase_keywords(
            processed_data['noun_phrases'], top_k=10
        )
        
        # Generate final recommendations
        summary['recommended_keywords'] = self._get_final_recommendations(summary, top_k)
        
        return summary
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _normalize_scores(self, keyword_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Normalize scores to 0-1 range"""
        if not keyword_scores:
            return []
        
        scores = [score for _, score in keyword_scores]
        max_score = max(scores)
        min_score = min(scores)
        
        if max_score == min_score:
            return [(kw, 1.0) for kw, _ in keyword_scores]
        
        normalized = []
        for keyword, score in keyword_scores:
            norm_score = (score - min_score) / (max_score - min_score)
            normalized.append((keyword, norm_score))
        
        return normalized
    
    def _is_stop_phrase(self, phrase: str) -> bool:
        """Check if phrase should be filtered out"""
        stop_phrases = {
            'et al', 'figure', 'table', 'section', 'equation', 'algorithm',
            'method', 'approach', 'result', 'conclusion', 'discussion',
            'paper', 'study', 'research', 'analysis', 'work', 'article'
        }
        return phrase.lower() in stop_phrases
    
    def _get_final_recommendations(self, summary: dict, top_k: int) -> List[Tuple[str, str]]:
        """
        Generate final keyword recommendations based on all methods
        
        Args:
            summary (dict): Summary from all extraction methods
            top_k (int): Number of recommendations
            
        Returns:
            List[Tuple[str, str]]: List of (keyword, source) tuples
        """
        # Collect all keywords with their sources
        all_keywords = {}
        
        # From combined (highest priority)
        for i, (kw, score) in enumerate(summary['combined_keywords'][:top_k]):
            if kw not in all_keywords:
                all_keywords[kw] = {'score': score, 'source': 'combined', 'rank': i}
        
        # From TF-IDF
        for i, (kw, score) in enumerate(summary['tfidf_keywords'][:top_k]):
            if kw not in all_keywords:
                all_keywords[kw] = {'score': score, 'source': 'tfidf', 'rank': i + 20}
        
        # From RAKE
        for i, (kw, score) in enumerate(summary['rake_keywords'][:top_k]):
            if kw not in all_keywords:
                all_keywords[kw] = {'score': score, 'source': 'rake', 'rank': i + 20}
        
        # Sort by rank and return
        sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1]['rank'])
        
        recommendations = []
        for kw, data in sorted_keywords[:top_k]:
            recommendations.append((kw, data['source']))
        
        return recommendations


if __name__ == "__main__":
    # Test the keyword extractor
    extractor = KeywordExtractor()
    
    sample_text = """
    Machine learning algorithms have revolutionized artificial intelligence.
    Deep learning neural networks show remarkable performance in computer vision.
    Natural language processing techniques enable advanced text analysis.
    Convolutional neural networks excel at image recognition tasks.
    """
    
    print("Testing keyword extraction...")
    
    # Test TF-IDF
    tfidf_keywords = extractor.extract_tfidf_keywords(sample_text)
    print(f"TF-IDF Keywords: {tfidf_keywords[:5]}")
    
    # Test RAKE
    rake_keywords = extractor.extract_rake_keywords(sample_text)
    print(f"RAKE Keywords: {rake_keywords[:5]}")
    
    # Test combined
    combined_keywords = extractor.extract_combined_keywords(sample_text, sample_text)
    print(f"Combined Keywords: {combined_keywords[:5]}")
