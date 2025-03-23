from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

# Caminho para o dataset
DATASET_PATH = "dataset_tjpe.jsonl"
MODEL_NAME = "microsoft/phi-2"

# Carregar dataset local
data = load_dataset("json", data_files={"train": DATASET_PATH})
# Renomeia a coluna "output" para "text" para que o trainer a reconheça
data["train"] = data["train"].rename_column("output", "text")

# Configuração do bitsandbytes para quantização 4-bit (supondo que seu ambiente CUDA 11.8 esteja configurado)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# Habilitar gradient checkpointing para reduzir memória
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# Configuração do LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Verificar quais parâmetros estão treináveis
trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
print("Parâmetros treináveis:", trainable_params)

# Se estiver usando gradient checkpointing, desative-o para testar
if hasattr(model, "gradient_checkpointing_disable"):
    model.gradient_checkpointing_disable()

def my_data_collator(features):
    texts = [feature["text"] for feature in features]
    batch = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )
    # Define os labels para que o modelo compute a loss
    batch["labels"] = batch["input_ids"].clone()
    return batch

# Argumentos de treinamento ajustados para menor uso de memória
training_args = TrainingArguments(
    output_dir="./modelo_phi2_tjpe",
    per_device_train_batch_size=1,  # menor batch size
    gradient_accumulation_steps=8,  # para manter um batch efetivo de 8
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    evaluation_strategy="no",
    remove_unused_columns=False,
    dataloader_num_workers=0
)

# Inicializar treinador com TRL
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    data_collator=my_data_collator,
)

trainer.train()
model.save_pretrained("./modelo_phi2_tjpe")

