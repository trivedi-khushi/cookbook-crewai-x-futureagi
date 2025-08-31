import os
from dotenv import load_dotenv
from futureagi import Evaluator

load_dotenv()
api_key = os.getenv("FUTURE_AGI_API_KEY")

evaluator = Evaluator(api_key=api_key)

def evaluate_text(text: str):
    result = evaluator.check(
        text=text,
        checks=["hallucination", "accuracy", "edge_case"]
    )
    return result
