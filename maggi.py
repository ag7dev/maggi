import os
import time
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from colorama import Fore, Style, init, Back

# Colorama initialization
init(autoreset=True)

# Configuration
MODEL_NAME = "gpt2"
INPUT_FOLDER = "input_folder"
MODEL_SAVE_PATH = "custom_model"
COMMAND_PREFIX = "/"
MAX_INPUT_LENGTH = 512
MAX_GENERATION_LENGTH = 200

class ColorPrinter:
    """Helper class for colored console output"""
    @staticmethod
    def print_error(msg: str) -> None:
        print(f"{Fore.BLACK}{Back.RED}[ERROR]{Style.RESET_ALL} {msg}")
    
    @staticmethod
    def print_warning(msg: str) -> None:
        print(f"{Fore.YELLOW}[!] {msg}{Style.RESET_ALL}")
    
    @staticmethod
    def print_success(msg: str) -> None:
        print(f"{Fore.GREEN}[✓] {msg}{Style.RESET_ALL}")
    
    @staticmethod
    def print_info(msg: str) -> None:
        print(f"{Fore.CYAN}[i] {msg}{Style.RESET_ALL}")
    
    @staticmethod
    def print_bold(msg: str) -> None:
        print(f"{Style.BRIGHT}{msg}{Style.RESET_ALL}")

def show_welcome_banner() -> None:
    """Displays welcome banner"""
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}")
    print(r"""

  __  __                   _ 
 |  \/  |                 (_)
 | \  / | __ _  __ _  __ _ _ 
 | |\/| |/ _` |/ _` |/ _` | |
 | |  | | (_| | (_| | (_| | |
 |_|  |_|\__,_|\__, |\__, |_|
                __/ | __/ |  
               |___/ |___/   

    """)
    print(f"{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Welcome to Maggi v1.0{Style.RESET_ALL}")
    print(f"{Fore.LIGHTBLACK_EX}Type '{COMMAND_PREFIX}help' for available commands\n")

def check_and_create_folders() -> None:
    """Verifies folder structure"""
    ColorPrinter.print_info("Checking folder structure...")
    
    if not os.path.exists(INPUT_FOLDER):
        ColorPrinter.print_error(f"Input folder '{INPUT_FOLDER}' not found!")
        os.makedirs(INPUT_FOLDER)
        ColorPrinter.print_success(f"Created '{INPUT_FOLDER}' directory")
        ColorPrinter.print_warning("Please add training data and restart!")
        exit(1)

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def load_texts_from_folder(folder_path: str) -> List[str]:
    """Loads text files from folder"""
    texts = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().strip()
                if content:
                    texts.append(content)
                else:
                    ColorPrinter.print_warning(f"Skipped empty file: {file_name}")
    return texts

def download_and_prepare_model():
    """Downloads and prepares the base model if not found."""
    # Check if model directory contains essential files
    required_files = ["config.json", "pytorch_model.bin", "vocab.json"]
    model_exists = all(os.path.exists(os.path.join(MODEL_SAVE_PATH, f)) for f in required_files)
    
    if not model_exists:
        ColorPrinter.print_warning("Base model not found, downloading...")
        try:
            # Create folder if not exists
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            model.resize_token_embeddings(len(tokenizer))
            
            tokenizer.save_pretrained(MODEL_SAVE_PATH)
            model.save_pretrained(MODEL_SAVE_PATH)
            ColorPrinter.print_success("Successfully downloaded and saved base model")
        except Exception as e:
            ColorPrinter.print_error(f"Download failed: {str(e)}")
            exit(1)
    else:
        ColorPrinter.print_info("Base model already present in directory")

def initialize_model() -> tuple:
    """Initializes tokenizer and model"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_SAVE_PATH)
        
        # Special tokens configuration
        if not tokenizer.pad_token:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            model.resize_token_embeddings(len(tokenizer))
        
        return tokenizer, model
    except Exception as e:
        ColorPrinter.print_error(f"Model initialization failed: {str(e)}")
        exit(1)

def validate_training_data(texts: List[str]) -> bool:
    """Validates if training data contains conversational patterns"""
    if len(texts) < 10:
        ColorPrinter.print_warning("Low training data (recommended: 10+ quality conversations)")
        return False
    
    conversation_count = sum(1 for text in texts if "\n" in text)  # Simple conversation detection
    if conversation_count < 5:
        ColorPrinter.print_warning("Few conversation-like structures detected")
    
    return True

def train_model() -> None:
    """Trains the model"""
    ColorPrinter.print_bold(f"\n{Fore.MAGENTA}=== TRAINING STARTED ==={Style.RESET_ALL}")
    start_time = time.time()
    
    try:
        texts = load_texts_from_folder(INPUT_FOLDER)
        if not texts:
            ColorPrinter.print_error("No training data found!")
            return

        if not validate_training_data(texts):
            ColorPrinter.print_warning("Training data quality might affect results")

        dataset = Dataset.from_dict({"text": texts})
        tokenizer, model = initialize_model()

        # Tokenization
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )

        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # Training configuration
        training_args = TrainingArguments(
            output_dir=MODEL_SAVE_PATH,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            logging_steps=100,
            save_steps=500,
            logging_dir='./logs',
            report_to="none"
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
        )

        # Training
        trainer.train()
        trainer.save_model(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)
        
        training_time = time.time() - start_time
        ColorPrinter.print_success(
            f"Training completed successfully in {training_time:.2f}s"
        )
        
        # After training verification
        ColorPrinter.print_info("Testing model with sample input...")
        test_input = "Hello"  # Simple test prompt
        inputs = tokenizer(test_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ColorPrinter.print_info(f"Sample response: {response}")
        
    except Exception as e:
        ColorPrinter.print_error(f"Training failed: {str(e)}")

def show_help() -> None:
    """Displays help information"""
    help_text = f"""
    {Fore.CYAN}Available commands:{Style.RESET_ALL}
    {COMMAND_PREFIX}help    - Show this help
    {COMMAND_PREFIX}exit    - Exit the program
    {COMMAND_PREFIX}retrain - Retrain the model
    """
    print(help_text)

def chat_with_model() -> None:
    """Main chat loop"""
    ColorPrinter.print_info("Loading model for conversation...")
    try:
        tokenizer, model = initialize_model()
    except Exception as e:
        ColorPrinter.print_error(f"Failed to load model: {str(e)}")
        return

    ColorPrinter.print_success("Ready for conversation!")
    print(f"\n{Fore.LIGHTBLACK_EX}Type your message or '{COMMAND_PREFIX}help' for commands\n")
    
    while True:
        try:
            user_input = input(f"{Fore.YELLOW}{Style.BRIGHT}You ➜ {Style.RESET_ALL}")
            
            if user_input.startswith(COMMAND_PREFIX):
                command = user_input[len(COMMAND_PREFIX):].strip().lower()
                if command == "exit":
                    break
                elif command == "retrain":
                    train_model()
                    tokenizer, model = initialize_model()
                elif command == "help":
                    show_help()
                else:
                    ColorPrinter.print_warning(f"Unknown command: '{command}'")
                continue

            # Input validation
            if not user_input.strip():
                continue
                
            # Response generation
            inputs = tokenizer(
                user_input,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_LENGTH
            )
            
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=MAX_GENERATION_LENGTH,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                temperature=0.7,
                do_sample=True
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n{Fore.GREEN}AI ➜ {Style.RESET_ALL}{response}\n")
            
        except KeyboardInterrupt:
            print("\n")
            ColorPrinter.print_warning("Conversation ended")
            break
        except Exception as e:
            ColorPrinter.print_error(f"Generation error: {str(e)}")

if __name__ == "__main__":
    show_welcome_banner()
    check_and_create_folders()  # Muss zuerst kommen!
    download_and_prepare_model()
    
    # Check if training is needed (modified logic)
    train_data_present = len(os.listdir(INPUT_FOLDER)) > 0
    model_files_present = len(os.listdir(MODEL_SAVE_PATH)) > 3  # Mindestens 3 Dateien
    
    if train_data_present and not model_files_present:
        ColorPrinter.print_warning("Training data found but no trained model - starting training!")
        train_model()
    
    chat_with_model()
