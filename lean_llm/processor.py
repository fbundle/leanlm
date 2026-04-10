type Language = str


class Processor:
    def marshal_input(self, input_text: str) -> Language:
        raise NotImplementedError

    def unmarshal_input(self, prompt: Language) -> str:
        raise NotImplementedError

    def unmarshal_output(self, completion: Language) -> str:
        raise NotImplementedError


class Qwen3Processor(Processor):
    def __init__(self):
        super().__init__()

    def marshal_input(self, input_text: str) -> Language:
        return "<|im_start|>user\n" + input_text + "<|im_end|>\n<|im_start|>assistant\n<think>\n"

    def unmarshal_input(self, prompt: Language) -> str:
        return prompt.lstrip("<|im_start|>user\n").rstrip("<|im_end|>\n<|im_start|>assistant\n<think>\n")

    def unmarshal_output(self, completion: Language) -> str:
        completion = completion.split("</think>")[-1]
        completion = completion.split("<|im_end|>")[0]
        return completion

class Gemma4Processor(Processor):
    def __init__(self):
        super().__init__()

    def marshal_input(self, input_text: str) -> Language:
        return "<bos><|turn>system\n<|think|><turn|>\n<|turn>user\n" + input_text + "<turn|>\n<|turn>model\n"

    def unmarshal_input(self, prompt: Language) -> str:
        return prompt.lstrip("<bos><|turn>system\n<|think|><turn|>\n<|turn>user\n").rstrip("<turn|>\n<|turn>model\n")

    def unmarshal_output(self, completion: Language) -> str:
        completion = completion.split("<channel|>")[-1]
        completion = completion.split("<turn|>")[0]
        return completion
