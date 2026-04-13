Language = str

class Processor(object):
    def marshal_input(self, input_text: str) -> Language:
        raise NotImplementedError

    def unmarshal_input(self, prompt: Language) -> str:
        raise NotImplementedError

    def unmarshal_output(self, completion: Language) -> tuple[str, str]:
        raise NotImplementedError

class Type1Processor(Processor):
    def __init__(
        self,
        bef_input: str,
        aft_input: str,
        end_reason: str,
        end_turn: str
    ) -> None:
        super().__init__()
        self.bef_input = bef_input
        self.aft_input = aft_input
        self.end_reason = end_reason
        self.end_turn = end_turn
    
    def marshal_input(self, input_text: str) -> Language:
        return self.bef_input + input_text + self.aft_input

    def unmarshal_input(self, prompt: Language) -> str:
        return prompt.lstrip(self.bef_input).rstrip(self.aft_input)

    def unmarshal_output(self, completion: Language) -> tuple[str, str]:
        # remove the first end_turn and everything after that
        completion = completion.split(self.end_turn)[0]
        # answer is after the last end_reason
        # reason is before the last end_reason
        parts = completion.split(self.end_reason)
        reason, answer = self.end_reason.join(parts[:-1]), parts[-1]
        return reason, answer





class Gemma4Processor(Processor):
    def __init__(self):
        super().__init__()

    def marshal_input(self, input_text: str) -> Language:
        return "<bos><|turn>system\n<|think|><turn|>\n<|turn>user\n" + input_text + "<turn|>\n<|turn>model\n"

    def unmarshal_input(self, prompt: Language) -> str:
        return prompt.lstrip("<bos><|turn>system\n<|think|><turn|>\n<|turn>user\n").rstrip("<turn|>\n<|turn>model\n")

    def unmarshal_output(self, completion: Language, reasoning: bool = False) -> str | tuple[str, str]:
        completion = completion.split("<channel|>")[-1]
        completion = completion.split("<turn|>")[0]
        return completion


class Qwen3Processor(Processor):
    def __init__(self):
        super().__init__()

    def marshal_input(self, input_text: str) -> Language:
        return "<|im_start|>user\n" + input_text + "<|im_end|>\n<|im_start|>assistant\n<think>\n"

    def unmarshal_input(self, prompt: Language) -> str:
        return prompt.lstrip("<|im_start|>user\n").rstrip("<|im_end|>\n<|im_start|>assistant\n<think>\n")

    def unmarshal_output(self, completion: Language, reasoning: bool = False) -> str | tuple[str, str]:
        completion = completion.split("</think>")[-1]
        completion = completion.split("<|im_end|>")[0]
        return completion

class DeepseekR1Processor(Processor):
    def __init__(self):
        super().__init__()

    def marshal_input(self, input_text: str) -> Language:
        return "<｜begin▁of▁sentence｜><｜User｜>" + input_text + "<｜Assistant｜><think>\n"

    def unmarshal_input(self, prompt: Language) -> str:
        return prompt.lstrip("<｜begin▁of▁sentence｜><｜User｜>").rstrip("<｜Assistant｜><think>\n")

    def unmarshal_output(self, completion: Language, reasoning: bool = False) -> str | tuple[str, str]:
        completion = completion.split("</think>")[-1]
        completion = completion.split("<｜end▁of▁sentence｜>")[0]
        return completion
