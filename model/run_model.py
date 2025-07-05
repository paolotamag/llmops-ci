from langfuse import observe, get_client

from model.modules.intent_classifier import IntentClassifier
from model.modules.customize_answer import CustomizeAnswer

@observe()
def run_model(customer_question : str):

    langfuse = get_client()
    langfuse.update_current_trace(name="Pet Food Customer Service")

    classifier = IntentClassifier()
    customizer = CustomizeAnswer()


    try:
        intent_classification_result = classifier.classify_intent(customer_question)

        customize_answer_result = customizer.customize_answer(
            conversation=customer_question,
            intent=intent_classification_result.intent_id,
        )



        return {
            "intent_id": intent_classification_result.intent_id,
            "intent_description": intent_classification_result.intent_description,
            "confidence": intent_classification_result.confidence,
            "reasoning": intent_classification_result.reasoning,
            "prepared_answer_id": customize_answer_result.prepared_answer_id,
            "prepared_answer_text": customize_answer_result.prepared_answer_text,
            "custom_answer": customize_answer_result.custom_answer
        }

        
    except Exception as e:
        print(f"Error processing question '{customer_question}': {e}")
