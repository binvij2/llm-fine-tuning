## 🧪 TinyLlama Fine-Tuning on Medical Q&A with PEFT (LoRA)

This code demonstrates how to fine-tune a quantized **TinyLlama** model using **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA**, trained on a dataset of medical/general health Q&A from StackExchange. The workflow includes:

- Preprocessing domain-specific Q&A pairs into instruction-following format
- Loading the base model in quantized 4-bit precision for memory efficiency
- Applying LoRA to inject trainable adapters into selected transformer layers
- Training with domain-specific text data
- Merging the LoRA weights back into the base model for export and deployment

### 📦 Required Libraries
Make sure the following are installed:

```bash
pip install torch transformers bs4 aiofiles bitsandbytes peft trl datasets matplotlib wandb
