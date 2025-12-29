import torch
import subprocess
import os
from pathlib import Path
from peft import PeftModel
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

# ===============================
# 2. CONFIG
# ===============================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Use absolute paths based on script location
BASE_DIR = Path(__file__).parent.parent / "model"
TRAIN_FILE = str(Path(__file__).parent / "dataset" / "training.jsonl")
TEST_FILE = str(Path(__file__).parent / "dataset" / "test.jsonl")
OUTPUT_DIR = str(BASE_DIR / "lora-qwen2.5-1.5b-final")
MERGED_DIR = str(BASE_DIR / "qwen_merged")
GGUF_FILE = str(BASE_DIR / "qwen2.5-1.5b-tour-assistant-q4.gguf")
TEMP_GGUF = str(BASE_DIR / "temp.gguf")

# Path to llama.cpp (adjust if needed)
LLAMA_CPP_DIR = Path(__file__).parent / "llama.cpp"

MAX_LENGTH = 512
BATCH_SIZE = 2  # Tăng nhẹ nếu VRAM cho phép
GRAD_ACCUM = 4
EPOCHS = 3
LR = 2e-4

# ===============================
# 3. TOKENIZER
# ===============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Trainer chuẩn thích padding phải

# ===============================
# 4. LOAD MODEL (4BIT)
# ===============================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

# ===============================
# 5. LORA CONFIG (TỐI ƯU CHO QWEN)
# ===============================
lora_config = LoraConfig(
    r=16,  # Tăng r lên chút để học tốt hơn
    lora_alpha=32,
    # Thêm các module gate_proj, up_proj, down_proj giúp Qwen thông minh hơn nhiều
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===============================
# 6. DATASET & PREPROCESS (QUAN TRỌNG)
# ===============================
dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "test": TEST_FILE})


def preprocess(example):
    # 1. Tạo Message chuẩn ChatML
    messages = [
        {
            "role": "system",
            "content": "Bạn là trợ lý hỗ trợ khách hàng. Hãy trả lời ngắn gọn và chính xác.",
        },
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]

    # 2. Format thành text có chứa <|im_start|>, <|im_end|>...
    # Quan trọng: add_generation_prompt=False để nó tự thêm EOS vào cuối
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # 3. Tokenize toàn bộ chuỗi
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,  # Để DataCollator xử lý padding động sẽ tiết kiệm bộ nhớ hơn
        add_special_tokens=False,
    )

    # 4. Tạo Masking (Chỉ tính loss cho phần Assistant trả lời)
    # Tìm vị trí bắt đầu câu trả lời của Assistant
    # Trong ChatML, phần trả lời bắt đầu sau header: "<|im_start|>assistant\n"
    input_ids = tokenized["input_ids"]
    labels = input_ids.copy()

    # Tính toán prompt (System + User) để gán -100
    # Mẹo: Tạo prompt giả để đo độ dài
    prompt_messages = messages[:-1]  # Bỏ phần assistant
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(
        prompt_text, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False
    )["input_ids"]
    prompt_len = len(prompt_ids)

    # Gán -100 cho phần Prompt
    for i in range(len(labels)):
        if i < prompt_len:
            labels[i] = -100
        else:
            labels[i] = input_ids[i]  # Giữ nguyên phần output để tính loss

    tokenized["labels"] = labels
    return tokenized


tokenized_dataset = dataset.map(
    preprocess, remove_columns=dataset["train"].column_names
)

# ===============================
# 7. TRAINER SETUP
# ===============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    fp16=True,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="none",
)

# Dùng DataCollatorForSeq2Seq để padding động (dynamic padding) -> Tiết kiệm VRAM và nhanh hơn
data_collator = DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

# ===============================
# 8. TRAIN & SAVE
# ===============================
print("Đang bắt đầu train...")
trainer.train()

print("Đang lưu LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ===============================
# 9. CLEANUP MEMORY BEFORE MERGE
# ===============================
print("Đang dọn dẹp bộ nhớ trước khi merge...")
del model
del trainer
torch.cuda.empty_cache()

# ===============================
# 10. MERGE LORA INTO BASE MODEL
# ===============================
print("Đang gộp LoRA vào base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cpu",  # Merge trên CPU để tiết kiệm VRAM
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

merged_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
merged_model = merged_model.merge_and_unload()  # Gộp hẳn vào

# Create output directory if not exists
os.makedirs(MERGED_DIR, exist_ok=True)

merged_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
print(f"Đã gộp xong model tại: {MERGED_DIR}")

# Cleanup merged model from memory
del base_model
del merged_model
torch.cuda.empty_cache()

# ===============================
# 11. CONVERT TO GGUF FORMAT
# ===============================
print("Đang chuyển đổi sang GGUF...")

convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
quantize_bin = LLAMA_CPP_DIR / "llama-quantize"

# Check if llama.cpp tools exist
if not convert_script.exists():
    print(f"⚠️  Không tìm thấy {convert_script}")
    print("Hãy clone llama.cpp và build trước:")
    print("  git clone https://github.com/ggerganov/llama.cpp")
    print("  cd llama.cpp && make")
else:
    try:
        # Step 1: Convert HF to GGUF (FP16)
        subprocess.run(
            [
                "python",
                str(convert_script),
                MERGED_DIR,
                "--outfile",
                TEMP_GGUF,
                "--outtype",
                "f16",
            ],
            check=True,
        )
        print(f"Đã convert sang GGUF FP16: {TEMP_GGUF}")

        # Step 2: Quantize to 4-bit (Q4_K_M - tốt cho inference)
        if quantize_bin.exists():
            subprocess.run(
                [
                    str(quantize_bin),
                    TEMP_GGUF,
                    GGUF_FILE,
                    "q4_k_m",
                ],
                check=True,
            )
            print(f"Đã nén xuống 4-bit: {GGUF_FILE}")

            # Clean up temp file
            if os.path.exists(TEMP_GGUF):
                os.remove(TEMP_GGUF)
        else:
            print(f"⚠️  Không tìm thấy {quantize_bin}")
            print("File GGUF FP16 đã có tại:", TEMP_GGUF)
            print("Hãy chạy quantize thủ công:")
            print(f"  ./llama.cpp/llama-quantize {TEMP_GGUF} {GGUF_FILE} q4_k_m")

    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi convert GGUF: {e}")
        print("Hãy kiểm tra llama.cpp đã được cài đặt đúng chưa.")

print("=" * 50)
print("🎉 HOÀN TẤT!")
print(f"   LoRA adapter: {OUTPUT_DIR}")
print(f"   Merged model: {MERGED_DIR}")
if os.path.exists(GGUF_FILE):
    print(f"   GGUF file:    {GGUF_FILE}")
print("=" * 50)
