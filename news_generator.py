import requests
import json
import os
import time

class NewsGenerator:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required. Please set your Groq API key.")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = model_name
        
        # Available models with their specifications
        self.available_models = {
            "llama-3.3-70b-versatile": {"name": "Llama 3.3 70B Versatile", "context": "128K", "speed": "Medium"},
            "gemma2-9b-it": {"name": "Gemma 2 9B IT", "context": "8K", "speed": "Fast"},
            "llama3-70b-8192": {"name": "Llama 3 70B", "context": "8K", "speed": "Medium"},
            "llama3-8b-8192": {"name": "Llama 3 8B", "context": "8K", "speed": "Very Fast"}
        }
        
        # Validate model
        if self.model not in self.available_models:
            raise ValueError(f"Model '{self.model}' not supported. Available models: {list(self.available_models.keys())}")
    
    def get_available_models(self):
        """Return list of available models with descriptions"""
        return self.available_models
    
    def set_model(self, model_name):
        """Change the current model"""
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not supported. Available models: {list(self.available_models.keys())}")
        self.model = model_name
        
    def _make_api_call(self, prompt):
        """Make API call to Groq"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return self._get_fallback_response(prompt)
                
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            return self._get_fallback_response(prompt)
        except Exception as e:
            print(f"Unexpected Error: {e}")
            return self._get_fallback_response(prompt)
    
    def _get_fallback_response(self, prompt):
        """Provide fallback response when API is unavailable"""
        if "bullet" in prompt.lower():
            return """• This is a fallback bullet-point summary
• Generated when the Groq API is unavailable
• Provides basic structure for testing
• Contains 5-8 key points as requested
• Maintains the required format"""
        elif "abstract" in prompt.lower():
            return "This is a fallback abstract summary generated when the Groq API is unavailable. It provides a scholarly-style overview of the article content, maintaining the required 80-120 word count and academic tone for testing purposes when the external service cannot be reached."
        else:
            return "This is a simple fallback summary for teens when the API is down. It explains the main points in easy words that anyone can understand quickly."
    
    def generate_summaries(self, article_text):
        """Generate all three types of summaries"""
        
        # Bullet-point summary prompt
        bullet_prompt = f"""
Generate a bullet-point summary of the following news article. 
Requirements:
- Provide 5-8 crisp bullet points
- Each bullet should be concise and informative
- Focus on the most important facts and developments
- Use bullet points (•) format

Article:
{article_text}

Bullet-point summary:
"""
        
        # Abstract summary prompt
        abstract_prompt = f"""
Generate an abstract summary of the following news article.
Requirements:
- Write in scholarly/academic style
- Use 80-120 words
- Be formal and objective
- Include key findings and implications

Article:
{article_text}

Abstract summary:
"""
        
        # Simple summary prompt
        simple_prompt = f"""
Generate a simple summary of the following news article.
Requirements:
- Write for teenagers (ages 13-18)
- Use 40-80 words
- Use simple, everyday language
- Make it easy to understand

Article:
{article_text}

Simple summary:
"""
        
        print(f"Generating bullet-point summary using {self.model}...")
        bullet_summary = self._make_api_call(bullet_prompt)
        
        # Add small delay between API calls
        time.sleep(0.5)
        
        print(f"Generating abstract summary using {self.model}...")
        abstract_summary = self._make_api_call(abstract_prompt)
        
        time.sleep(0.5)
        
        print(f"Generating simple summary using {self.model}...")
        simple_summary = self._make_api_call(simple_prompt)
        
        return {
            'bullet': bullet_summary,
            'abstract': abstract_summary,
            'simple': simple_summary
        }
