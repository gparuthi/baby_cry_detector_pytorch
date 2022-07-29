from dataclasses import dataclass
import os
from typing import Dict

import numpy as np
import librosa
import requests
import torch


def record_sound_to_file(fn: str, device: int):
    """Uses ffmpeg to record sound to a file"""
    os.system(f"ffmpeg -loglevel error -f avfoundation -i ':{device}' -t 5 {fn} -y")


def get_detection(fn: str, model, indtocat: Dict[int, str], device: str):
    """Gets detection for a sound file"""
    spec = FeatureExtractor.spec_to_image(FeatureExtractor.get_melspectrogram_db(fn))
    spec_t = torch.tensor(spec).to(device, dtype=torch.float32)
    pr = model.forward(spec_t.reshape(1, 1, *spec_t.shape))
    all_confidences = pr.cpu().detach().numpy()
    conf, ind = np.max(all_confidences), np.argmax(all_confidences)
    return conf, indtocat[ind]


class FeatureExtractor:
    @staticmethod
    def get_melspectrogram_db(
        file_path,
        sr=None,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmin=20,
        fmax=8300,
        top_db=80,
    ):
        """Returns a melspectrogram for a sound file"""
        wav, sr = librosa.load(file_path, sr=sr)
        if wav.shape[0] < 5 * sr:
            wav = np.pad(wav, int(np.ceil((5 * sr - wav.shape[0]) / 2)), mode="reflect")
        else:
            wav = wav[: 5 * sr]
        spec = librosa.feature.melspectrogram(
            y=wav,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )
        spec_db = librosa.power_to_db(spec, top_db=top_db)
        return spec_db

    @staticmethod
    def spec_to_image(spec, eps=1e-6):
        """Converts melspectrogram to image"""
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        assert spec_min and spec_max
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        return spec_scaled


@dataclass
class TelegramClient:
    bot_key: str
    chat_id: str

    def send_audio(self, file_path: str):
        """Send a audio file to a telegram app chat via the BOT API"""
        url = (
            f"https://api.telegram.org/{self.bot_key}/sendAudio?chat_id={self.chat_id}"
        )
        with open(file_path, "rb") as f:
            response = requests.post(url, files={"audio": f})
            print(response.text)

    def send_msg(self, msg: str):
        url = f"https://api.telegram.org/{self.bot_key}/sendMessage?chat_id={self.chat_id}&text={msg}"
        response = requests.post(url)
        print(response.text)


def get_logger():
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger()
