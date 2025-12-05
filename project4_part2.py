import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import  AdamW

# part 2 of the project, using and finetuning the same datasett as part1
def load_dataset(path):
    # reads an entire text file and returns it as one string
    # 'path' should be the path to the dataset file (so like 'data.txt' or 'data2.txt').
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def finetune_hf(dataset_path, save_path, model_name='distilgpt2', epochs=8, lr=5e-5, max_length=128):
    #finetunes a small HuggingFace model on a text dataset
    #loads the model and tokenizer, tokenizes the text, trains for a couple of epochs,
    #and saves the model for later use
    device = torch.device('cpu')

    print('Loading pretrained model and tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.train()

    text = load_dataset(dataset_path)

    print('Tokenizing dataset...')
    encodings = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    input_ids = encodings.input_ids.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    print(f'Starting finetuning for {epochs} epoch(s)...')

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}: Loss = {loss.item():.4f}')

    print(f'Saving finetuned model to {save_path}...')
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print('Done.')


def generate_hf(prompt, model_path, max_new_tokens=50):
    #generates text from a finetuned HuggingFace model
    #loads the model and tokenizer from tthe folder, feeds in a prompt,
    #and returns the text that the model thinks comes next
    device = torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.9,
        top_p=0.95
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    #commandline interface for part 2
    #lets you either
    #type  1 finetune a pretrained HuggingFace model on your dataset
    #type  2 generate text using a previously finetuned model
    #type  exit to quit the program
    while True:
        print('\n--- PART 2: HuggingFace Finetuning ---')
        print('[1] Finetune pretrained model')
        print('[2] Generate text using finetuned model')
        print('[Exit] Quit')

        choice = input('> ').strip().lower()

        if choice == 'exit':
            break

        elif choice == '1':
            dataset = input('Dataset path (e.g., data2.txt): ').strip()
            if not os.path.isfile(dataset):
                print("Error: Dataset file not found.")
                continue
            save_path = input('Save directory for finetuned model: ').strip()
            try:
                    os.makedirs(save_path)
            except Exception as e:
                print(f"Error: Could not create directory '{save_path}'. {e}")
                continue
            try:
                finetune_hf(dataset, save_path)
            except Exception as e:
                print("Finetuning failed:", e)

        elif choice == '2':
            model_path = input('Model directory: ').strip()
            if not os.path.isdir(model_path):
                print("Error: Model directory not found.")
                continue
            prompt = input('Prompt: ')
            try:
                print(generate_hf(prompt, model_path))
            except Exception as e:
                print("Generation failed:", e)

########
# references
########
# https://chatgpt.com/share/693199a4-68f4-8002-81c1-bd88d0aedd93
#   > some questions about how do i even do this
#
# https://chatgpt.com/share/693197b4-2eb4-8002-af39-3c599dca906d
#   > fixing errors, the first question was me being a stupid and being 1 folder out by accident
# 