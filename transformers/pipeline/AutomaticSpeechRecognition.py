from transformers import pipeline

transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small", device=1)
trans = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")

print(trans)
