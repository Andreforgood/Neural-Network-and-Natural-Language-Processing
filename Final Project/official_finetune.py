# ======= final_train.py =======
import os, gc, json
from collections import Counter
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    BlipProcessor, BlipForQuestionAnswering,
    AdamW, get_scheduler
)

# --------- 环境与路径 ----------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"   # 6 张卡
DEVICE = torch.device("cuda")

LOCAL_BLIP = "/home/guest_a/yuezhao/dongwenou/NLP_Final/blip-vqa-base"
ROOT      = "/home/guest_a/yuezhao/dongwenou/NLP_Final"
TRAIN_IMG = f"{ROOT}/train2014"
VAL_IMG   = f"{ROOT}/val2014"
TRAIN_Q   = f"{ROOT}/v2_OpenEnded_mscoco_train2014_questions.json"
TRAIN_A   = f"{ROOT}/v2_mscoco_train2014_annotations.json"
VAL_Q     = f"{ROOT}/v2_OpenEnded_mscoco_val2014_questions.json"
VAL_A     = f"{ROOT}/v2_mscoco_val2014_annotations.json"

BEST_LR   = 3.5887266e-05
BEST_WD   = 0.056132
BEST_DROPOUT = 0.236              # 新增 dropout
WARMUP_RATIO = 0.0786              # 新增 warmup ratio
FREEZE_V  = False
FREEZE_T  = False                # 若不冻结 text，设为 False
BATCH     = 128
EPOCHS    = 2
SAVE_PATH = "./blip_vqa_final3.pt"

# --------- 数据工具 ----------
def load_json(path, key): 
    with open(path) as f: return json.load(f)[key]

def id2name(img_id, prefix): 
    return f"{prefix}_{img_id:012d}.jpg"

def build_ds(q_path, a_path, img_dir, prefix):
    qs  = load_json(q_path, "questions")
    anns = {a["question_id"]: a for a in load_json(a_path, "annotations")}
    data = []
    for q in qs:
        ann = anns.get(q["question_id"]); 
        if not ann: continue
        answers = [x["answer"].lower().strip() for x in ann["answers"]]
        ans = Counter(answers).most_common(1)[0][0]
        img = os.path.join(img_dir, id2name(q["image_id"], prefix))
        data.append((img, q["question"], ans))
    return data

class VQADataset(Dataset):
    def __init__(self, data, proc, max_len=32):
        self.data, self.proc, self.max_len = data, proc, max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img, q, a = self.data[idx]
        x = self.proc(Image.open(img).convert("RGB"), q, text_target=a,
                      padding="max_length", truncation=True,
                      max_length=self.max_len, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in x.items()}

# ---------- 准备数据 ----------
print("loading processor & dataset …")
proc  = BlipProcessor.from_pretrained(LOCAL_BLIP)
train = build_ds(TRAIN_Q, TRAIN_A, TRAIN_IMG, "COCO_train2014")
val   = build_ds(VAL_Q,   VAL_A,   VAL_IMG,   "COCO_val2014")
train_loader = DataLoader(VQADataset(train, proc), BATCH, shuffle=True)
val_loader   = DataLoader(VQADataset(val,   proc), BATCH, shuffle=False)
print(f"train={len(train)},  val={len(val)}")

# ---------- 初始化模型 ----------
from transformers import BlipConfig
cfg = BlipForQuestionAnswering.from_pretrained(LOCAL_BLIP).config
cfg.hidden_dropout_prob = BEST_DROPOUT
cfg.attention_probs_dropout_prob = BEST_DROPOUT
model = BlipForQuestionAnswering.from_pretrained(LOCAL_BLIP, config=cfg)

if FREEZE_V:
    for p in model.vision_model.parameters(): p.requires_grad = False
if FREEZE_T:
    for p in model.text_encoder.parameters(): p.requires_grad = False

model = torch.nn.DataParallel(model).to(DEVICE)

# ---------- 优化器 & scheduler ----------
opt = AdamW(model.parameters(), lr=BEST_LR, weight_decay=BEST_WD)
steps = len(train_loader) * EPOCHS
sched = get_scheduler("linear", opt, num_warmup_steps=int(WARMUP_RATIO * steps),
                      num_training_steps=steps)

# ---------- 验证函数 ----------
@torch.no_grad()
def val_loss():
    model.eval(); s = 0
    for batch in tqdm(val_loader):
        batch = {k:v.to(DEVICE) for k,v in batch.items()}
        loss = model(**batch).loss.mean().item()
        print('val_loss is:', loss)
        s += loss
    model.train(); return s/len(val_loader)

# ---------- 训练 ----------
best = float("inf")
for ep in range(EPOCHS):
    loop = tqdm(train_loader, desc=f"Epoch {ep+1}/{EPOCHS}")
    for batch in loop:
        batch = {k:v.to(DEVICE) for k,v in batch.items()}
        loss = model(**batch).loss.mean()
        loss.backward(); opt.step(); sched.step(); opt.zero_grad()
        loop.set_postfix(loss=f"{loss.item():.4f}")
    v = val_loss(); print(f"val_loss={v:.4f}")
    if v < best:
        best = v
        torch.save(model.module.state_dict(), SAVE_PATH)
        print(f"saved best → {SAVE_PATH}")

# ---------- END ----------