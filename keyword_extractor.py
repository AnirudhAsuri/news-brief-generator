import nltk
from collections import Counter
import re
import string
import spacy
from typing import List, Dict, Tuple

class KeywordExtractor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        
        # Initialize NLTK components
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add common words to stop words
        self.stop_words.update(['said', 'say', 'says', 'according', 'also', 'would', 'could', 'should'])
        
        # Initialize spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_available = True
            print("spaCy model loaded successfully")
        except OSError:
            print("spaCy model not found. Falling back to NLTK-only extraction.")
            self.nlp = None
            self.spacy_available = False
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove bullet points and special characters at the start of lines
        text = re.sub(r'^[•\-\*]\s*', '', text, flags=re.MULTILINE)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except periods and commas
        text = re.sub(r'[^\w\s\.\,]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text, max_keywords=15, include_entities=True):
        """Extract keywords from text using NLTK and optionally spaCy for NER"""
        try:
            # Try enhanced extraction with spaCy if available
            if self.spacy_available and include_entities:
                return self._extract_keywords_with_ner(text, max_keywords)
            else:
                return self._extract_keywords_nltk_only(text, max_keywords)
        
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            # Fallback: simple word extraction
            return self._simple_keyword_extraction(text, max_keywords)
    
    def _extract_keywords_with_ner(self, text, max_keywords=15):
        """Enhanced keyword extraction using both NLTK and spaCy NER"""
        from nltk.tokenize import word_tokenize
        from nltk import pos_tag
        
        # Process with spaCy for NER
        if self.nlp is None:
            return self._extract_keywords_nltk_only(text, max_keywords)
        
        doc = self.nlp(text)
        
        # Extract named entities
        entities = []
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
                # Clean and add entity
                entity_text = ent.text.strip().lower()
                if len(entity_text) > 2 and entity_text.replace(' ', '').isalpha():
                    entities.append(entity_text)
        
        # NLTK-based keyword extraction (as before)
        cleaned_text = self.preprocess_text(text)
        tokens = word_tokenize(cleaned_text)
        
        filtered_tokens = [
            token for token in tokens 
            if (token.lower() not in self.stop_words and 
                len(token) > 2 and 
                token.isalpha())
        ]
        
        pos_tags = pos_tag(filtered_tokens)
        important_tokens = [
            word for word, pos in pos_tags 
            if pos.startswith(('NN', 'JJ', 'VB'))
        ]
        
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in important_tokens]
        
        # Combine NLTK keywords and entities
        all_keywords = lemmatized_tokens + entities
        
        # Count frequency and get most common
        word_freq = Counter(all_keywords)
        
        # Prioritize entities by giving them extra weight
        for entity in entities:
            if entity in word_freq:
                word_freq[entity] += 1  # Boost entity scores
        
        # Get top keywords
        top_keywords = [word for word, freq in word_freq.most_common(max_keywords)]
        
        return top_keywords
    
    def _extract_keywords_nltk_only(self, text, max_keywords=15):
        """Extract keywords using only NLTK (fallback method)"""
        from nltk.tokenize import word_tokenize
        from nltk import pos_tag
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Tokenize
        tokens = word_tokenize(cleaned_text)
        
        # Filter tokens: remove stopwords, short words, and non-alphabetic tokens
        filtered_tokens = [
            token for token in tokens 
            if (token.lower() not in self.stop_words and 
                len(token) > 2 and 
                token.isalpha())
        ]
        
        # POS tagging - keep only nouns, adjectives, and verbs
        pos_tags = pos_tag(filtered_tokens)
        important_tokens = [
            word for word, pos in pos_tags 
            if pos.startswith(('NN', 'JJ', 'VB'))
        ]
        
        # Lemmatize
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in important_tokens]
        
        # Count frequency and get most common
        word_freq = Counter(lemmatized_tokens)
        
        # Get top keywords
        top_keywords = [word for word, freq in word_freq.most_common(max_keywords)]
        
        return top_keywords
    
    def extract_entities(self, text) -> List[Dict[str, str]]:
        """Extract named entities using spaCy"""
        if not self.spacy_available or self.nlp is None:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            # Entity label descriptions
            label_descriptions = {
                'PERSON': 'People, including fictional',
                'ORG': 'Companies, agencies, institutions',
                'GPE': 'Countries, cities, states',
                'PRODUCT': 'Objects, vehicles, foods, etc.',
                'EVENT': 'Named hurricanes, battles, wars, sports events',
                'WORK_OF_ART': 'Titles of books, songs, etc.',
                'LAW': 'Named documents made into laws',
                'LANGUAGE': 'Any named language'
            }
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': label_descriptions.get(ent.label_, ent.label_)
                })
            
            return entities
        
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []
    
    def _simple_keyword_extraction(self, text, max_keywords):
        """Fallback keyword extraction without advanced NLP"""
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Remove common stop words manually
        basic_stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        filtered_words = [word for word in words if word not in basic_stop_words and len(word) > 2]
        
        # Count and return most common
        word_freq = Counter(filtered_words)
        return [word for word, freq in word_freq.most_common(max_keywords)]
