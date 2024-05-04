from time import sleep
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

load_dotenv()

client = OpenAI()

file = client.files.create(
    file=Path("conversations.jsonl"),
    purpose="fine-tune",
)
print(file)
print("File created!")

fine_tune = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-3.5-turbo",
    hyperparameters={
        "n_epochs": 2
    }
)
print(fine_tune)
print('Fine-Tuning Started ...')

while True:
    res = client.fine_tuning.jobs.retrieve(fine_tune.id)
    if res.finished_at is not None:
        break
    else:
        print("......")
        sleep(10)
print(res)
print("Model Fine-Tuned Successfully!")
