# Example 4.24
# Question Answering menggunakan Transformers

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model_name = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

question = "What is the name of the company?"
context = "We created Biox Systems Ltd company back in the year of 2000."

inputs = tokenizer(
    question,
    context,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

start_index = torch.argmax(outputs.start_logits)
end_index = torch.argmax(outputs.end_logits)

answer = tokenizer.decode(
    inputs["input_ids"][0][start_index:end_index + 1]
)

print("Question:", question)
print("Context:", context)
print("Answer:", answer)