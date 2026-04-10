from llm_trainer.trainer import Processor, Language


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

def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()