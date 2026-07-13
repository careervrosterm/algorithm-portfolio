import pandas as pd
import json
import validator
import classifier
import logging

def main():
    logging.basicConfig(filename='data/myapp.log', level=logging.INFO)
    logging.info("Beginning process.")
    df_requests = pd.read_csv('data/requests.csv')
    logging.info("CSV loaded. Sending requests to OpenAI.")
    response_list = []
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
    df_responses.to_csv('\\data\\responses.csv')
    return df_responses

if __name__ == "__main__":
    main()