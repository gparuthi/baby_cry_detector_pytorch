Yet another hack for detecting baby cries using Pytorch. Intended for learning purposes.

### Dependencies:

- FFMPEG
- Pytorch, librosa (see [requirements.txt](requirements.txt))

## Training

Follow the [python notebook](train/ESC50_Pytorch.ipynb). Tested on Google Colab.

## Detection:

1. Update `config.py`

   - Find your microphone device

   ```
   ffmpeg -f avfoundation -list_devices true -i ""
   ```
 
   - Add telegram details for getting notifications if needed

2. Run

```
python run.py
```

## References:

1. Original training script: https://github.com/hasithsura/Environmental-Sound-Classification
