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


class Gemma4Processor(Type1Processor):
    def __init__(self):
        super().__init__(
            bef_input="<bos><|turn>system\n<|think|><turn|>\n<|turn>user\n",
            aft_input="<turn|>\n<|turn>model\n",
            end_reason="<channel|>",
            end_turn="<turn|>",
        )


class Qwen3Processor(Type1Processor):
    def __init__(self):
        super().__init__(
            bef_input="<|im_start|>user\n",
            aft_input="<|im_end|>\n<|im_start|>assistant\n<think>\n",
            end_reason="</think>",
            end_turn="<|im_end|>",
        )

class Qwen3PhoenixProcessor(Type1Processor):
    def __init__(self):
        super().__init__(
            bef_input="",
            aft_input="",
            end_reason="</think>",
            end_turn="<|im_end|>",
        )

class DeepseekR1Processor(Type1Processor):
    def __init__(self):
        super().__init__(
            bef_input="<｜begin▁of▁sentence｜><｜User｜>",
            aft_input="<｜Assistant｜><think>\n",
            end_reason="</think>",
            end_turn="<｜end▁of▁sentence｜>",
        )
