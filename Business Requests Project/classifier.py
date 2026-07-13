from openai import OpenAI
import json
from validator import output_schema

client = OpenAI(
  api_key="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"
)

def classify_ai(client: str, request:str): 
    prompt = f"""
    You are helping classify workflow requests a variety of different company clients.

    Client: {client}
    Request: {request}

    Return a structured classification using the exact JSON structure: 
    {json.dump(output_schema)}

    Some additional rules:
    1. Do not produce fictional information.
    2. If there is not enough info posed to assess priority, use "Unknown".
    3. If there is not enough info posed to assess risk, use "Unknown".
    4. If not enough context is given to assess risk and priority, or if there is no actionable goal that can be processed, then needs_human_review = True.
    """
    response = client.responses.parse(
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
        ],
        text=output_schema
    )

    return response.output_parsed