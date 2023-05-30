## the code needs ffmpeg, use apt-get install ffmpeg to install

from transformers import pipeline

classifier = pipeline(task="audio-classification", model="superb/hubert-base-superb-er", device=0)
preds = classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
print(preds)
