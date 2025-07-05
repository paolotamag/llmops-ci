import json
# from langfuse import openai

from langfuse import openai
from pydantic import BaseModel
from typing import List, Optional, Dict
import os

class IntentClassification(BaseModel):
    intent_id: str
    intent_description: str
    confidence: float
    reasoning: str

class IntentClassifier:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPEN_AI_KEY'))
        self.intents = self._load_intents("example_data/intents_pet_food.json")
        
    def _load_intents(self, file_path: str) -> Dict[str, str]:
        """Load available intents from JSON file"""
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Intents file not found at {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {file_path}")
    
    def classify_intent(self, customer_question: str) -> IntentClassification:
        """Classify customer question into one of the available intents"""
        
        # Format intents for the prompt
        intents_list = "\n".join([f"ID {key}: {description}" for key, description in self.intents.items()])
        
        # Create system prompt
        system_prompt = f"""
        You are an AI assistant for a dog food company's customer service department. 
        Your task is to analyze customer questions and classify them into the most appropriate intent category.
        
        Available intent categories:
        {intents_list}
        
        For each customer question, you must:
        1. Select the most appropriate intent ID from the available categories
        2. Return the corresponding intent description
        3. Provide a confidence score (0.0 to 1.0)
        4. Explain your reasoning for the classification
        
        Be precise and consider the context of dog food, pet nutrition, orders, shipping, and customer support.
        You must return the intent_id as one of the available IDs: {list(self.intents.keys())}
        
        Respond in the following JSON format:
        {{
            "intent_id": "selected_intent_id",
            "intent_description": "corresponding_description",
            "confidence": 0.95,
            "reasoning": "explanation_for_selection"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                name="intent_classifier",
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Classify this customer question: {customer_question}"}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            result_json = json.loads(response.choices[0].message.content)
            
            # Create IntentClassification object
            result = IntentClassification(**result_json)
            
            # Validate that the returned intent_id exists in our intents
            if result.intent_id not in self.intents:
                raise ValueError(f"Invalid intent_id returned: {result.intent_id}")
            
            return result
            
        except Exception as e:
            raise Exception(f"Error during intent classification: {str(e)}")
