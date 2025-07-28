
import wandb
wandb.login() # needs api key

import torch
import json
import asyncio
import aiofiles
from datasets import load_dataset
from bs4 import BeautifulSoup
from functools import partial
from transformers import  AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import LoraConfig, get_peft_model , prepare_model_for_kbit_training, PeftModel
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer

# sync helper functions
def convert_to_chatml_format(qa_pair):
    return {
        "text": f"<|system|>\nYou are a AI Medical Advisor and you advice on health queries</s>\n \
        <|user|>\n{qa_pair['question']}</s>\n \
        <|assistant|>\n{qa_pair['best_answer']}</s>\n"
    }


def clean_html_preserve_links(html_text):
    soup = BeautifulSoup(html_text, "html.parser")

    # Convert <a href="...">link text</a> to [link text](url)
    for a_tag in soup.find_all("a", href=True):
        link_text = a_tag.get_text(strip=True)
        link_url = a_tag['href']
        markdown_link = f"[{link_text}]({link_url})"
        a_tag.replace_with(markdown_link)

    # Remove all other tags and get text
    return soup.get_text(separator="\n", strip=True)


# async helper functions
async def run_in_executor(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(func, *args))


async def process_post(post, sem):
    async with sem:
        qa_answers = post.get("answers", [])
        if not qa_answers:
            return None

        question = await run_in_executor(clean_html_preserve_links, post.get("question_body", ""))

        best = max(qa_answers, key=lambda x: x.get("score", 0))
        low = min(qa_answers, key=lambda x: x.get("score", 0))

        high_score = best.get("score", 0)
        lowest_score = low.get("score", 0)

        best_answer = await run_in_executor(clean_html_preserve_links, best.get("body", ""))
        lowest_answer = await run_in_executor(clean_html_preserve_links, low.get("body", ""))

        return {
            "question": question,
            "best_answer": best_answer,
            "high_score": high_score,
            "lowest_answer": lowest_answer,
            "lowest_score": lowest_score
        }

async def write_jsonl_async(data, filename):
    async with aiofiles.open(filename, mode="w") as f:
        for qa in data:
            chatml = convert_to_chatml_format(qa)
            await f.write(json.dumps(chatml, ensure_ascii=False) + "\n")



async def main(ds, output_file="medical_chatml.jsonl", max_concurrent=100):
    sem = asyncio.Semaphore(max_concurrent)
    tasks = [process_post(post, sem) for post in ds]
    results = await asyncio.gather(*tasks)
    qa_set = [r for r in results if r is not None]
    await write_jsonl_async(qa_set, output_file)
    print(f"Wrote {len(qa_set)} items to {output_file}")

import nest_asyncio
nest_asyncio.apply()

ds = load_dataset("ymoslem/MedicalSciences-StackExchange")["train"]
await main(ds, output_file="medical_chatml.jsonl", max_concurrent=100)

# Quantization Configuration
checkpoint = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_dir ="./results/peft-qlora-sml-training"
dataset = load_dataset("json", data_files="medical_chatml.jsonl")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# Load the model with quantization
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False  # Disable cache for training
model.config.pretraining_tp = 1  # Set pretraining tensor parallelism to 1
# Load the LLama Tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = "<PAD>"
tokenizer.padding_side = "left"
  # Set pad token to eos token

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
        per_device_train_batch_size=10,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        # optim="paged_adamw_32bit",
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        logging_steps=10,
        fp16=True,
        gradient_checkpointing=False,
        report_to="wandb" # Weights & Biases logging
)
# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    args=training_args,
    peft_config=peft_config,
    )

trained_model_adapter_name = "TinyLlama-1.1B-Chat-v1.0-peft-medical-sciences-adapter"
trained_model_merged = "TinyLlama-1.1B-Chat-v1.0-peft-medical-sciences-merged"

trainer.train()
trainer.model.save_pretrained(trained_model_adapter_name)

base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    low_cpu_mem_usage=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(
    base_model,
    trained_model_adapter_name,
    low_cpu_mem_usage=True,
    device_map="auto",
)

merged_model = model.merge_and_unload()
merged_model.save_pretrained(trained_model_merged)
tokenizer.save_pretrained(trained_model_merged)