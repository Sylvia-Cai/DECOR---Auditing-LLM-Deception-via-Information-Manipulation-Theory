from llm import TargetLLM
import requests
import json
from openai import OpenAI

class ChatGPT(TargetLLM):
    # your api_key
    api_key=""
    
    def __init__(self, mode='gpt-3.5-turbo', secret=api_key, debug=False):
        self.api_key = secret
        self.debug = debug
        self.url = "https://api.aiproxy.io/v1"
        modes_dict = {
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "gpt-4o": "gpt-4o",
            "gpt-4": "gpt-4"
        }
        self.model_key = modes_dict[mode]
        print(self.model_key)
        # exit()
        self.version = mode
        
    def chat(self, messages)->str:
        client = OpenAI(
            # change to the aiproxy secret api_key here 
            api_key=self.api_key,
            # change to the aiproxy url here
            base_url=self.url
        )
        try:
            response = client.chat.completions.create(
                model=self.model_key,
                messages = messages
            )
            if(self.debug):
                print(response)
            response = response.choices[0].message.content
            return response
        except Exception as e:
            return f'[error] {e}'

    def info(self):
        return {
            "company": 'OpenAI',
            "version": self.version
        }
    
if __name__ == '__main__':
    llm = ChatGPT(mode='gpt-4o')
    ans = llm.chat('hello')
    print(ans)
