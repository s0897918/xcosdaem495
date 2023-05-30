from transformers import pipeline

question_answerer = pipeline(task="question-answering", framework='pt', device=0)
preds = question_answerer(
    question="What is the name of the repository?",
    context="Company name is AMD, but The name of the repository is huggingface/transformers",
)
print(
    f"score: {round(preds['score'], 4)}, start: {preds['start']}, end: {preds['end']}, answer: {preds['answer']}"
)
