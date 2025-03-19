"""
Functions to send messages and images to a telegram chat. (Used just to monitor asyncronously the process of gs)
"""

import os
from dotenv import load_dotenv
load_dotenv()
import requests

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

print(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

def send_tg_msg(bot_message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage?"

    text = "[㎖] >> " + bot_message

    out = requests.post(url=url, params={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    if out.status_code != 200:
        print(f"Error sending telegram message: {out.status_code}")


def send_tg_img(img_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto?"
    files = {"photo": open(img_path, "rb")}
    data = {"chat_id": TELEGRAM_CHAT_ID}
    out = requests.post(url=url, files=files, data=data)
    if out.status_code != 200:
        print(f"Error sending telegram image: {out.status_code}")