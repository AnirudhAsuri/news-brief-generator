from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimilarityCalculator:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            max_features=1000
        )
    
    def compute_cosine_similarity(self, text1, text2):
        """Compute cosine similarity using TF-IDF vectors"""
        try:
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            
            # Compute cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Return similarity between the two texts
            return similarity_matrix[0, 1]
        
        except Exception as e:
            print(f"Error computing cosine similarity: {e}")
            return 0.0
    
    def compute_jaccard_similarity(self, keywords1, keywords2):
        """Compute Jaccard similarity between two sets of keywords"""
        try:
            # Convert to sets for intersection and union operations
            set1 = set(keyword.lower() for keyword in keywords1)
            set2 = set(keyword.lower() for keyword in keywords2)
            
            # Handle empty sets
            if len(set1) == 0 and len(set2) == 0:
                return 1.0
            
            # Calculate Jaccard similarity
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            
            if len(union) == 0:
                return 0.0
            
            return len(intersection) / len(union)
        
        except Exception as e:
            print(f"Error computing Jaccard similarity: {e}")
            return 0.0
    
    def compute_similarity(self, original_text, summary_text, original_keywords, summary_keywords, cosine_weight=0.7, jaccard_weight=0.3):
        """Compute combined similarity score with customizable weights"""
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = cosine_weight + jaccard_weight
        if total_weight > 0:
            cosine_weight = cosine_weight / total_weight
            jaccard_weight = jaccard_weight / total_weight
        else:
            # Default weights if both are zero
            cosine_weight = 0.7
            jaccard_weight = 0.3
        
        # Calculate cosine similarity
        cosine_sim = self.compute_cosine_similarity(original_text, summary_text)
        
        # Calculate Jaccard similarity
        jaccard_sim = self.compute_jaccard_similarity(original_keywords, summary_keywords)
        
        # Combined score with custom weights
        combined_score = cosine_weight * cosine_sim + jaccard_weight * jaccard_sim
        
        return {
            'cosine': float(cosine_sim),
            'jaccard': float(jaccard_sim),
            'combined': float(combined_score),
            'weights': {
                'cosine_weight': float(cosine_weight),
                'jaccard_weight': float(jaccard_weight)
            }
        }
