import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

def send_mesage(text: str, token: str | None = None, chat: str | None = None):
    if token is None:
        token = os.environ.get("TG_TOKEN", default=None)
    if chat is None:
        chat = os.environ.get("TG_CHAT", default=None)
    
    if token is None or chat is None:
        raise RuntimeError("token and chat must be set")
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat,
        "text": text,
    }

    res = requests.post(url, json=payload)
    res.raise_for_status()
    return res

if __name__ == "__main__":
    for text in sys.argv[1:]:
        send_mesage(text)