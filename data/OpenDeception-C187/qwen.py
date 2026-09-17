from llm import TargetLLM
import dashscope
from http import HTTPStatus

class Qwen(TargetLLM):
    """
    @intro: 通义千问是阿里云自主研发的超大规模语言模型，它结合了先进的自然语言处理技术和大规模数据训练，具备广泛的知识和普适性，能够理解和回答来自不同领域的各种问题。
    """
    # your api_key
    api_key=""

    def __init__(self, mode='qwen2-7b-instruct', secret=api_key, debug=False):
        self.secret = secret
        self.debug = debug
        modes_dict = {
            "qwen2-7b-instruct": "qwen2-7b-instruct",
            "qwen2-72b-instruct": "qwen2-72b-instruct",
            "qwen2.5-3b-instruct":"qwen2.5-3b-instruct",
            "qwen2.5-7b-instruct":"qwen2.5-7b-instruct",
            "qwen2.5-14b-instruct":"qwen2.5-14b-instruct",
            "qwen2.5-32b-instruct":"qwen2.5-32b-instruct",
            "qwen2.5-72b-instruct":"qwen2.5-72b-instruct"
        }
        dashscope.api_key=secret
        self.model_key = modes_dict[mode]
        print(self.model_key)
        self.version = mode
     
    def chat(self, messages)->str:
        response = dashscope.Generation.call(
            model=self.model_key,
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
            "company": '通义千问',
            "version": self.version,
        }
    
if __name__ == '__main__':
    llm = Qwen()
    ans = llm.chat('你好')
    print(ans)
