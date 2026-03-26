from __future__ import annotations


from typing import Iterator, Any
import os


from threading import Thread
from pydantic import BaseModel
from transformers import TextIteratorStreamer


class Message(BaseModel):
    role: str
    content: str

class Streamer:
    def chat(self, message_list: list[Message]) -> Iterator[str]:
        raise NotImplemented

class TransformerStreamer(Streamer):
    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        generate_kwargs: dict | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.generate_kwargs = {}
        if generate_kwargs is not None:
            self.generate_kwargs.update(generate_kwargs)
    
    def _generate(self, message_list: list[Message], text_streamer: TextIteratorStreamer):
        input_text = self.tokenizer.apply_chat_template(
            conversation=[message.model_dump() for message in message_list],
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(input_text, return_tensors="pt")
        model_inputs = model_inputs.to(self.model.device).to(self.model.dtype)
        self.model.generate(
            **model_inputs,
            streamer=text_streamer,
            **self.generate_kwargs,
        )
    
    def chat(self, message_list: list[Message]) -> Iterator[str]:
        text_streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # skip the prompt, stream the output only
            skip_special_tokens=True,  # pass into tokenizer.decode, skip EOS for example
        )
        thread = Thread(target=TransformerStreamer._generate, args=(self, message_list, text_streamer))
        thread.start()

        def streamer() -> Iterator[str]:
            yield from text_streamer
            thread.join()

        return streamer()

class Conversation:
    def __init__(self, path: str = "conversation.jsonl"):
        self.path = path
        self.message_list = []
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path))
            open(path, mode="w").close()
        
        with open(path) as f:
            for line in f:
                message = Message.model_validate_json(line)
                self.message_list.append(message)
    
    def append(self, role: str, content: str) -> Conversation:
        message = Message(
            role=role, 
            content=content,
        )
        self.message_list.append(message)
        with open(self.path, mode="w") as f:
            f.write(message.model_dump_json() + "\n")
        
        return self

    def append_user_message(self, content: str) -> Conversation:
        return self.append(role="user", content=content)

    def append_assistant_message(self, content: str) -> Conversation:
        return self.append(role="assistant", content=content)
    
    def append_system_message(self, content: str) -> Conversation:
        return self.append(role="system", content=content)