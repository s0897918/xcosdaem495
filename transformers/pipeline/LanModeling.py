from transformers import pipeline

prompt = "Hugging Face is a community-based open-source platform for machine learning."
generator = pipeline(task="text-generation", device=0)
print(generator(prompt))  # doctest: +SKIP

text = "Hugging Face is a community-based open-source <mask> for machine learning."
fill_mask = pipeline(task="fill-mask", device=0)
preds = fill_mask(text, top_k=1)
preds = [
    {
        "score": round(pred["score"], 4),
        "token": pred["token"],
        "token_str": pred["token_str"],
        "sequence": pred["sequence"],
    }
    for pred in preds
]
print (preds)
