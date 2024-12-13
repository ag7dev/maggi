import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from colorama import Fore, Style, init

# Initialize colorama for cross-platform compatibility
init(autoreset=True)

# Configurations
MODEL_NAME = "gpt2"  # Base model (can be adjusted)
INPUT_FOLDER = "input_folder"  # Folder with your text data
MODEL_SAVE_PATH = "custom_model"  # Path to save the trained model
COMMAND_PREFIX = "/"

def check_and_create_folders():
    """Checks if the required folders exist, creates them if not."""
    if not os.path.exists(INPUT_FOLDER):
        print(f"{Fore.RED}[ERROR] Input folder '{INPUT_FOLDER}' does not exist. Please create it.")
        exit(1)
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"{Fore.RED}[ERROR] Model save path '{MODEL_SAVE_PATH}' does not exist. Creating...")
        os.makedirs(MODEL_SAVE_PATH)
        print(f"{Fore.GREEN}[INFO] Created '{MODEL_SAVE_PATH}' directory.")

def load_texts_from_folder(folder_path):
    """Loads all text data from a folder."""
    texts = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                texts.append(file.read())
    return texts

def download_and_prepare_model():
    """Downloads and prepares the base model if not found."""
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"{Fore.YELLOW}[INFO] Base model not found. Downloading...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})  # Add pad_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.resize_token_embeddings(len(tokenizer))  # Resize tokenizer to model
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)
        model.save_pretrained(MODEL_SAVE_PATH)
        print(f"{Fore.GREEN}[SUCCESS] Base model downloaded and saved.")
    else:
        print(f"{Fore.GREEN}[INFO] Base model already exists.")

def train_model():
    """Trains the model with text data from the input folder."""
    print(f"{Fore.CYAN}[INFO] Training started...")

    # Load and combine text data
    texts = load_texts_from_folder(INPUT_FOLDER)
    if not texts:
        print(f"{Fore.RED}[ERROR] No text data found. Please add files to the input folder.")
        return

    dataset = Dataset.from_dict({"text": texts})

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})  # Add pad_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_SAVE_PATH)
    model.resize_token_embeddings(len(tokenizer))  # Resize tokenizer to model

    # Tokenization function
    def tokenize_function(examples):
        encoding = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        encoding["labels"] = encoding["input_ids"]  # Set target values for training
        return encoding

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    # Train the model
    trainer.train()
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"{Fore.GREEN}[SUCCESS] Training completed and model saved.")

def chat_with_model():
    """Interacts with the trained model."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_SAVE_PATH)

    print(f"{Fore.CYAN}[INFO] Chat started! Type your text (\"/exit\" to exit):")
    while True:
        user_input = input(f"{Fore.YELLOW}You: ")

        if user_input.startswith(COMMAND_PREFIX):
            command = user_input[len(COMMAND_PREFIX):].strip()
            if command == "exit":
                print(f"{Fore.GREEN}[INFO] Chat ended.")
                break
            elif command == "retrain":
                train_model()
                print(f"{Fore.GREEN}[INFO] Model retrained.")
            else:
                print(f"{Fore.RED}[ERROR] Unknown command: {command}")
            continue

        # Tokenize the input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        attention_mask = inputs['attention_mask']

        # Generate model response
        outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=200, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=2)

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"{Fore.GREEN}AI: {response}")


if __name__ == "__main__":
    check_and_create_folders()  # Check if folders exist
    download_and_prepare_model()  # Prepare base model
    train_model()  # Train the model
    chat_with_model()  # Start the chat interaction
