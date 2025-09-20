# Overview

This is a News Brief Generator application built with Streamlit that creates AI-powered summaries of news articles and evaluates their quality through similarity analysis. The application takes news articles as input and generates three different types of summaries (bullet-point, abstract, and simple-English), then uses machine learning techniques to assess the quality and similarity of each summary compared to the original article.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
The application uses **Streamlit** as the web framework, providing a simple and interactive user interface. The main application logic is contained in `app.py`, which handles user input, displays results, and orchestrates the various components of the system.

## Core Components
The system follows a **modular architecture** with separate classes for different functionalities:

- **NewsGenerator**: Handles AI-powered text summarization using the Groq API
- **SimilarityCalculator**: Computes similarity metrics between original articles and summaries
- **KeywordExtractor**: Extracts and processes keywords from text using NLTK

## AI Integration
The application integrates with **Groq's LLaMA model** (llama3-8b-8192) for text generation. The system is designed with fallback mechanisms to handle API failures gracefully, ensuring the application remains functional even when external services are unavailable.

## Text Processing Pipeline
The system implements a comprehensive text analysis pipeline:

1. **Text Preprocessing**: Cleaning and normalizing input text
2. **Summary Generation**: Creating three distinct summary types (bullet-point, abstract, simple-English)
3. **Keyword Extraction**: Using NLTK for tokenization, stop-word removal, and lemmatization
4. **Similarity Analysis**: Computing cosine similarity using TF-IDF vectors and Jaccard similarity using keyword overlap
5. **Quality Assessment**: Combining multiple similarity metrics with weighted scoring (70% cosine + 30% Jaccard)

## Data Processing
The application uses **scikit-learn** for TF-IDF vectorization and cosine similarity calculations, and **NLTK** for natural language processing tasks including tokenization, POS tagging, and lemmatization.

# External Dependencies

## AI Services
- **Groq API**: Primary service for text summarization using LLaMA models
- **API Endpoint**: https://api.groq.com/openai/v1/chat/completions

## Python Libraries
- **Streamlit**: Web application framework for the user interface
- **scikit-learn**: Machine learning library for TF-IDF vectorization and similarity calculations
- **NLTK**: Natural language processing toolkit for text preprocessing and keyword extraction
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing support
- **requests**: HTTP client for API communication

## Environment Configuration
- **GROQ_API_KEY**: Environment variable for secure API key storage
- **NLTK Data**: Automatic download of required language models and corpora (punkt, stopwords, averaged_perceptron_tagger, wordnet)