import json
from typing import Dict

import torch

from utils import (
    TelegramClient,
    get_detection,
    get_logger,
    record_sound_to_file,
)
from config import (
    MODEL_PATH,
    CATEGORY_INDEX_MAP_PATH,
    DEVICE,
    FFMPEG_AUDIO_DEVICE,
    TELEGRAM_BOT_KEY,
    TELEGRAM_CHAT_ID,
)

logger = get_logger()

telegram_client = TelegramClient(TELEGRAM_BOT_KEY, TELEGRAM_CHAT_ID)


def start():
    resnet_model, indtocat = get_model()
    while True:
        try:
            run_detection_and_notify(resnet_model, indtocat)
        except Exception as e:
            print(e)


def get_model():
    with open(CATEGORY_INDEX_MAP_PATH) as f:
        indtocat: Dict[str, str] = json.load(f)
        indtocat: Dict[int, str] = {int(k): v for k, v in indtocat.items()}
        logger.info(indtocat)

    with open(MODEL_PATH, "rb") as f:
        resnet_model = torch.load(f, map_location=DEVICE)
    return resnet_model, indtocat


def run_detection_and_notify(resnet_model, indtocat: Dict[int, str]):
    fn = "temp.wav"
    record_sound_to_file(fn, device=FFMPEG_AUDIO_DEVICE)
    confidence, detection = get_detection(fn, resnet_model, indtocat, DEVICE)
    logger.info(f"{detection}-{confidence}")

    if "baby" in detection:
        telegram_client.send_msg(f"{detection}_{confidence}")
        record_sound_to_file("temp.mp3", device=FFMPEG_AUDIO_DEVICE)
        telegram_client.send_audio("temp.mp3")


if __name__ == "__main__":
    start()
