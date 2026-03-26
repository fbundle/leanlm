from __future__ import annotations

import os


from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class MessageList:
    file_path: str
    message_list: list[Message]

    def __init__(self, file_path: str):
        self.file_path = file_path

        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path))
            open(file_path, mode="w").close()

        with open(file_path) as f:
            self.message_list = [Message.model_validate_json(line) for line in f]

    def append_message(self, role: str, content: str) -> MessageList:
        message = Message(
            role=role,
            content=content,
        )
        self.message_list.append(message)
        open(self.file_path, mode="a").write(message.model_dump_json())

        return self

    def append_user_message(self, content: str) -> MessageList:
        return self.append_message(role="user", content=content)

    def append_assistant_message(self, content: str) -> MessageList:
        return self.append_message(role="assistant", content=content)

    def append_system_message(self, content: str) -> MessageList:
        return self.append_message(role="system", content=content)
