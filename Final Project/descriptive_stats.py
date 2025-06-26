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


def view_image_questions_answers(image_id, questions, annotations):
    """
    打印指定 image_id 的所有 (问题, 答案)。
    """
    print(f"Image ID: {image_id}")
    count = 0
    for q in questions:
        if q["image_id"] == image_id:
            qid = q["question_id"]
            q_text = q["question"]
            ans = next((a["answers"] for a in annotations if a["question_id"] == qid), [])
            ans_list = [a["answer"] for a in ans]
            print(f"\nQ{count+1}: {q_text}")
            print(f"A : {ans_list}")
            count += 1
    if count == 0:
        print("No questions found for this image_id.")
        

#questions = load_questions(TRAIN_Q_PATH)
#annotations = load_annotations(TRAIN_A_PATH)

#view_image_questions_answers(262175, questions, annotations)

questions = load_questions(VAL_Q_PATH)
annotations = load_annotations(VAL_A_PATH)

view_image_questions_answers(393284, questions, annotations)