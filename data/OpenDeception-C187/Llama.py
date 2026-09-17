from http import HTTPStatus
import os
import dashscope
from llm import TargetLLM

class Llama(TargetLLM):
    # your api_key
    api_key=""

    def __init__(self, mode='llama-3.1-70B-Instruct', secret=api_key, debug=False):# li
        
        self.secret = secret
        self.debug = debug
        modes_dict = {
            "llama3.1-8b-instruct":"llama3.1-8b-instruct",
            "llama3.1-70b-instruct":"llama3.1-70b-instruct",
            "llama3.1-405b-instruct":"llama3.1-405b-instruct"
        }
        dashscope.api_key=secret
        self.model_key = modes_dict.get(mode, modes_dict['llama3.1-405b-instruct'])
        self.version = mode    
        print(self.version)
        # exit()
                    
    def chat(self, messages)->str: 
        response = dashscope.Generation.call(
        model=self.model_key,
        # prompt=prompt
        messages = messages
        )
        try:
            if response.status_code == HTTPStatus.OK:
                if(self.debug):
                    print(response.output)  # The output text
                    print(response.usage)  # The usage information
                out = response.output['text']
                if(self.debug):
                    print(out)
                return out
            else:
                if(self.debug):
                    print(response.code)  # The error code.
                    print(response.message)  # The error message.
                return f"[error] {response.message}"
        except Exception as e:
            return f'[error] {e}'   
        
    def info(self):
        return {
        "company": 'Meta',
        "version": self.version,
    }

if __name__ == '__main__':
    llm = Llama(mode='llama3.1-70b-instruct')
    ans = llm.chat('hello')
    print(ans)
