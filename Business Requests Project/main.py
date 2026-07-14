import pandas as pd
import json
import validator
import classifier
import logging
from dotenv import load_dotenv
import os 

load_dotenv()
API_KEY = os.environ.get('API_KEY')

def main():
    logging.basicConfig(filename='Business Requests Project\\data\\app.log', level=logging.INFO)
    logging.info("Beginning process.")
    df_requests = pd.read_csv('Business Requests Project\\data\\requests.csv')
    logging.info("CSV loaded. Sending requests to OpenAI.")
    response_list = []
    if not API_KEY:
        with open("Business Requests Project\\data\\dummy.json", "r") as j:
            response =j.read()
        response = validator.validate(response)
        df_responses = pd.json_normalize(response)
        df_responses.to_csv('Business Requests Project\\data\\responses.csv')
        return df_responses

    for idx in df_requests.index:
        client = df_requests.loc[idx, "client"]
        request = df_requests.loc[idx, "request"]
        
        try:
            response = classifier.classify_ai(client, request)
            response = validator.validate(json.load(response))

            response["client"] = client
            response["request"] = request

            response_list.append(response)

        except Exception as e:
            logging.error(
                f"Failed to process request from {client}: {e}"
            )

    df_responses = pd.DataFrame(response_list)
    logging.info("Finished sending requests to OpenAI. Now reading responses into CSV format.")
    df_responses.to_csv('Business Requests Project\\data\\responses.csv')
    return df_responses

if __name__ == "__main__":
    main()