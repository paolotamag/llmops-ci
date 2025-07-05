from langfuse import openai
import os
from pydantic import BaseModel
from typing import List, Optional, Dict
import json

class AnswerCustomization(BaseModel):
    prepared_answer_id: str
    prepared_answer_text: str
    custom_answer: str

class CustomizeAnswer:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPEN_AI_KEY'))
        self.answers = self._load_answers("example_data/prepared_answers_pet_food.json")
        
    def _load_answers(self, file_path: str) -> Dict[str, str]:
        """Load available answers from JSON file"""
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Intents file not found at {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {file_path}")

    def customize_answer(self, conversation: str, intent: str) -> AnswerCustomization:
        """
        Rewrite a prepared answer in a restyled manner using customer and pet names extracted from the conversation.
        
        Args:
            conversation: The full conversation context
            intent: The intent key to find the prepared answer
        
        Returns:
            AnswerCustomization object with prepared answer and customized version
        """

        prepared_answer = self.answers.get(intent, None)

        if not prepared_answer:
            return AnswerCustomization(
                prepared_answer_id=intent,
                prepared_answer_text="",
                custom_answer=f"Error: No prepared answer found for intent '{intent}'"
            )

        
        # System prompt for extraction and restyling
        system_prompt = """You are an expert at rewriting customer service responses in a personalized, friendly manner.

        Your task is to:
        1. Extract the customer's name and pet's name from the conversation (if mentioned)
        2. Restyle the prepared answer to be more personal and engaging using those names
        
        Instructions:
        - Look for names mentioned in the conversation
        - Use the customer's name when available to make it personal
        - Include the pet's name when mentioned to create connection
        - Maintain the core information from the original answer
        - Make the tone warm and conversational
        - Return ONLY the restyled answer, nothing else"""
        
        # User prompt with conversation and prepared answer
        user_prompt = f"""
        Conversation:
        {conversation}
        
        Prepared answer to restyle:
        {prepared_answer}
        
        Please extract any customer and pet names from the conversation above, then restyle the prepared answer to be more personal and engaging using those names when available.
        """
        
        try:
            response = self.client.chat.completions.create(
                name="customize_answer",
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            custom_answer = response.choices[0].message.content.strip()
            
            return AnswerCustomization(
                prepared_answer_id=intent,
                prepared_answer_text=prepared_answer,
                custom_answer=custom_answer
            )
        
        except Exception as e:
            return AnswerCustomization(
                prepared_answer_id=intent,
                prepared_answer_text=prepared_answer,
                custom_answer=f"Error restyling answer: {str(e)}"
            )