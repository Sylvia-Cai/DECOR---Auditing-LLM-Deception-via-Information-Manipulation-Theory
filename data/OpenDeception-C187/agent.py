import time
from com_generate import chat_with_history

class Agent(object):
    def __init__(self, llm, role_profile='You are a helpful assistant.'):
        self.llm = llm
        print(self.llm)
        self.role_profile = role_profile
        self.conv_history = []
        self.conv_history.append({"role":"system", "content":role_profile})
        self.memory = []

    def process_message(self, msg):
        self.conv_history.append({"role":"user", "content": msg})
        answer = self.llm.chat(self.conv_history)
        while(answer == '[error]'):
            answer = chat_with_history(self.conv_history)
        self.conv_history.append({"role":"assistant", 'content':answer})
        return self.conv_history[-1]['content']
        
    
