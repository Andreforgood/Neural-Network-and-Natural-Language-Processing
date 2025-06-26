import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
import json
import os
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering, AdamW, get_scheduler
import optuna
from tqdm import tqdm

# ========== 基本路径设置 ==========
LOCAL_BLIP_PATH = "/home/guest_a/yuezhao/dongwenou/NLP_Final/blip-vqa-base"
TRAIN_IMAGE_DIR = "/home/guest_a/yuezhao/dongwenou/NLP_Final/train2014"
VAL_IMAGE_DIR = "/home/guest_a/yuezhao/dongwenou/NLP_Final/val2014"
TRAIN_Q_PATH = "/home/guest_a/yuezhao/dongwenou/NLP_Final/v2_OpenEnded_mscoco_train2014_questions.json"
TRAIN_A_PATH = "/home/guest_a/yuezhao/dongwenou/NLP_Final/v2_mscoco_train2014_annotations.json"
VAL_Q_PATH = "/home/guest_a/yuezhao/dongwenou/NLP_Final/v2_OpenEnded_mscoco_val2014_questions.json"
VAL_A_PATH = "/home/guest_a/yuezhao/dongwenou/NLP_Final/v2_mscoco_val2014_annotations.json"

# ========== 数据预处理函数 ==========
def load_questions(path):
    with open(path, 'r') as f:
        return json.load(f)['questions']

def load_annotations(path):
    with open(path, 'r') as f:
        return json.load(f)['annotations']

def image_id_to_filename(image_id, prefix="COCO_train2014"):
    return f"{prefix}_{image_id:012d}.jpg"

def create_dataset(questions, annotations, image_dir, prefix="COCO_train2014"):
    ann_dict = {ann['question_id']: ann for ann in annotations}
    dataset = []
    for q in questions:
        qid = q['question_id']
        if qid not in ann_dict:
            continue
        ann = ann_dict[qid]
        answers = [a['answer'].lower().strip() for a in ann['answers']]
        most_common = Counter(answers).most_common(1)[0][0]
        img_file = os.path.join(image_dir, image_id_to_filename(q['image_id'], prefix))
        dataset.append((img_file, q['question'], most_common))
    return dataset

# ========== 自定义 Dataset ==========
class VQADataset(Dataset):
    def __init__(self, data, processor, max_length=32):
        self.data = data
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, question, answer = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, question, text_target=answer,
                                padding="max_length", max_length=self.max_length,
                                truncation=True, return_tensors="pt")
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": inputs["labels"].squeeze(0),
        }

# ========== 验证 Loss ==========
from tqdm import tqdm

def evaluate_val_loss(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            pixel_values=pixel_values, labels=labels)
            print('batch val_loss:', outputs.loss.mean().item())
            total_loss += outputs.loss.mean().item()
    model.train()
    return total_loss / len(val_loader)

# ========== 预加载数据集 ==========
print("Loading data and processor...")
processor = BlipProcessor.from_pretrained(LOCAL_BLIP_PATH)
train_data = create_dataset(load_questions(TRAIN_Q_PATH), load_annotations(TRAIN_A_PATH), TRAIN_IMAGE_DIR)
train_data = train_data[:100000]  # 只用前10万个样本来调参
val_data = create_dataset(load_questions(VAL_Q_PATH), load_annotations(VAL_A_PATH), VAL_IMAGE_DIR, prefix="COCO_val2014")
val_data = val_data[:60000]
print(len(train_data))
print(len(val_data))
train_dataset = VQADataset(train_data, processor)
val_dataset = VQADataset(val_data, processor)

# ========== Optuna 调参函数 ==========
def objective(trial):
    import torch
    import gc
    from transformers import BlipConfig

    device = torch.device("cuda")

    # 超参数采样
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    warmup_ratio = trial.suggest_float("lr_scheduler_warmup_ratio", 0.05, 0.2)
    batch_size = 128
    freeze_vision = False
    freeze_text = False

    # 打印当前 Trial 的超参数组合
    print(f"\n Trial {trial.number} 超参数:")
    print(f"   - Learning rate: {lr}")
    print(f"   - Weight decay: {weight_decay}")
    print(f"   - Dropout rate: {dropout_rate}")
    print(f"   - Warmup ratio: {warmup_ratio}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Freeze vision: {freeze_vision}")
    print(f"   - Freeze text: {freeze_text}")

    # 修改 config 设置 dropout
    config = BlipForQuestionAnswering.from_pretrained(LOCAL_BLIP_PATH).config
    config.hidden_dropout_prob = dropout_rate
    config.attention_probs_dropout_prob = dropout_rate

    model = BlipForQuestionAnswering.from_pretrained(LOCAL_BLIP_PATH, config=config).to(device)

    if freeze_vision:
        for param in model.vision_model.parameters():
            param.requires_grad = False
    if freeze_text:
        for param in model.text_encoder.parameters():
            param.requires_grad = False

    model = torch.nn.DataParallel(model)
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    total_steps = len(train_loader) * 1  # 只跑一轮
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=warmup_steps,
                              num_training_steps=total_steps)

    model.train()
    for epoch in range(1):
        loop = tqdm(train_loader, desc=f"[Trial {trial.number}] Epoch {epoch+1}/1", leave=False)
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )

            loss = outputs.loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())

    val_loss = evaluate_val_loss(model, val_loader, device)
    print(f"[Trial {trial.number}] Validation Loss: {val_loss:.4f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return val_loss

# ========== 启动调参 ==========
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, show_progress_bar=True)

    print("最佳超参数：")
    print(study.best_trial.params)