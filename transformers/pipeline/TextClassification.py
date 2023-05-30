from transformers import pipeline

classifier = pipeline(task="sentiment-analysis", device=1)
preds = classifier("Hugging Face is the best thing since sliced bread!")
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]

print (preds)
