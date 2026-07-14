from openai import OpenAI
import json
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.environ.get('API_KEY')

openai_client = OpenAI(api_key=API_KEY)

prompt_schema = {
    "category": ["Communications", "Operations", "Finances", "Tech Issues", "Customer", "Data", "Documents", "Other"],
    "priority": ["Low", "Medium", "High", "Unknown"],
    "risk": ["Low", "Medium", "High", "Unknown"],
    "suggested_automation": "string",
    "reasoning": "string",
    "needs_human_review": "boolean"
}

def classify_ai(client: str, request:str):  
    with open("Business Requests Project\\data\\examples.json") as j:
        examples = json.load(j)

    prompt = f"""
    You are helping classify workflow requests a variety of different company clients.

    Here is an example list of correct outputs: {json.dumps(examples, indent=4)}

    Client: {client}
    Request: {request}

    Return a structured classification using the exact JSON structure: 
    {json.dumps(prompt_schema)}

    Some additional rules:
    1. Do not produce fictional information.
    2. If there is not enough info posed to assess priority, use "Unknown".
    3. If there is not enough info posed to assess risk, use "Unknown".
    4. If not enough context is given to assess risk and priority, or if there is no actionable goal that can be processed, then needs_human_review = True.
    """
    response = openai_client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": "You classify business workflow automation requests. Be concise and practical."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.output_text