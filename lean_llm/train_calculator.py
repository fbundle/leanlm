Language = str

class Processor(object):
    def marshal_input(self, input_text: str) -> Language:
        raise NotImplementedError

    def unmarshal_input(self, prompt: Language) -> str:
        raise NotImplementedError

    def unmarshal_output(self, completion: Language) -> str:
        raise NotImplementedError

class TrainConfig(BaseModel):
    prepare: bool = True

    output_dir: str
    processor: Processor
    tokenizer: Any # TODO - change to something that has .encode and .decode
    model: Any # TODO - change to something that has .generate

    batch_size: int
    accumulation_steps: int = 1

    max_completion_length: int
    temperature: float
    top_p: float
    min_p: float
    top_k: int

    repetition_penalty: float

    num_generations: int

    save_steps: int
    train_size: int
    eval_size: int

    deepspeed: str = "conf/ds_zero2.json"

def train(config: TrainConfig):
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    model, tokenizer = config.model, config.tokenizer

    if config.prepare:
        def apply_chat_template(*args, **kwargs):
            raise RuntimeError("GRPO must not use apply_chat_template")

        # prevent TRL from using apply_chat_template
        tokenizer.apply_chat_template = apply_chat_template

        def prepare_generate(generate):
            def helper(*args, **kwargs):
                if "min_new_tokens" not in kwargs:
                    kwargs["min_new_tokens"] = config.max_completion_length
                generate(*args, **kwargs)
            return helper

        # in prepare mode, always generate in full
        model.generate = prepare_generate(model.generate)

        # in prepare mode, train for 2 accumulation steps
        config.train_size = 2 * config.batch_size * config.accumulation_steps









