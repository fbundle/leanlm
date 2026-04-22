from pydantic import BaseModel

# THIS SHOULD BE BUILTIN IN TOKENIZER BUT NOONE BOTHER TO DO IT
# check tokenizer.apply_chat_template and tokenizer.parse_response


Language = str

class Processor(object):
    def append_system_input(self, prompt: Language) -> str:
        raise NotImplementedError

    def append_user_input(self, prompt: Language) -> str:
        raise NotImplementedError

    def parse_agent_output(self, completion: Language) -> tuple[str, str]:
        raise NotImplementedError

class Type1ProcessorConfig(BaseModel):
    prefix_system: str
    suffix_system: str
    prefix_user: str
    suffix_user: str
    begin_answer: str
    end_answer: str

class Type1Processor(Processor):
    def __init__(self, config: Type1ProcessorConfig) -> None:
        super().__init__()
        self.config = config
    
    def append_system_input(self, prompt: Language) -> str:
        return self.config.prefix_system + prompt + self.config.suffix_system

    def append_user_input(self, prompt: str) -> Language:
        return self.config.prefix_user + prompt + self.config.suffix_user

    def parse_agent_output(self, completion: Language) -> tuple[str, str]:
        # format
        # reasoning <begin_answer> answer <end_answer> rubbish
        completion = completion.split(self.config.end_answer)[0]
        chunks = completion.split(self.config.begin_answer)
        reason = self.config.begin_answer.join(chunks[:-1])
        answer = chunks[-1]
        return reason, answer
