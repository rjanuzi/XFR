import json
import logging
import traceback

from telegram import Bot, InputFile, ParseMode

_TELEGRAM_BOT_KEY = "1754067352:AAHgzr2qSoT7mqkGUw0nT01B2KM3ISHSTkE"  # rjanuzi_bot
_TELEGRAM_CHAT_ID = "-436350789"


def send_simple_message(text, bot_key=_TELEGRAM_BOT_KEY, chat_id=_TELEGRAM_CHAT_ID):
    try:
        result = Bot(token=bot_key).send_message(
            chat_id=chat_id, text=text, parse_mode=ParseMode.HTML
        )
        return True, result
    except:
        logging.error(traceback.format_exc())
        return False, traceback.format_exc()


def send_img(
    bot_key=_TELEGRAM_BOT_KEY,
    chat_id=_TELEGRAM_CHAT_ID,
    img_path=None,
    img_binary=None,
    caption="",
):
    assert img_path or img_binary
    try:
        img_data = (
            InputFile(img_binary) if img_binary else InputFile(open(img_path, "rb"))
        )
        result = Bot(token=bot_key).send_photo(
            chat_id=chat_id, photo=img_data, caption=caption
        )
        return True, result
    except:
        logging.error(traceback.format_exc())
        return False, traceback.format_exc()


def send_document(
    document_path, bot_key=_TELEGRAM_BOT_KEY, chat_id=_TELEGRAM_CHAT_ID, caption=""
):
    try:
        result = Bot(token=bot_key).send_document(
            chat_id=chat_id,
            document=InputFile(open(document_path, "rb")),
            caption=caption,
        )
        return True, result
    except:
        logging.error(traceback.format_exc())
        return False, traceback.format_exc()


def get_messages(bot_key):
    # Get offset
    try:
        offset_file = open(r"telegram_messages_offset.json", "r")
        offset = json.load(fp=offset_file)
        offset_file.close()
    except FileNotFoundError:
        offset = {"offset": 0}

    bot = Bot(token=bot_key)
    updates = bot.get_updates(offset=offset["offset"])
    updates = list(filter(lambda u: u and u.message, updates))  # Eliminates None
    messages = [
        {
            "chat_id": u.message.chat_id,
            "user_id": u.message.from_user.id,
            "message": u.message.text,
        }
        for u in updates
    ]

    if len(updates) > 0:
        offset["offset"] = updates[-1].update_id + 1

    offset_file = open(r"telegram_messages_offset.json", "w", encoding="utf-8")
    json.dump(obj=offset, fp=offset_file)
    offset_file.close()

    return messages
