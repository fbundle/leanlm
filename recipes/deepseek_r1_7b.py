from llm_trainer.trainer import Processor, Language

class DeepseekR1Processor(Processor):
    def __init__(self):
        super().__init__()

    def marshal_input(self, input_text: str) -> Language:
        return "<｜begin▁of▁sentence｜><｜User｜>" + input_text + "<｜Assistant｜><think>\n"

    def unmarshal_input(self, prompt: Language) -> str:
        return prompt.lstrip("<｜begin▁of▁sentence｜><｜User｜>").rstrip("<｜Assistant｜><think>\n")

    def unmarshal_output(self, completion: Language) -> str:
        completion = completion.split("</think>")[-1]
        completion = completion.split("<｜end▁of▁sentence｜>")[0]
        return completion



def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()