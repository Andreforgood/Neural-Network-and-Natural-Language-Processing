import os, random, argparse, json, math, cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
from transformers import BlipProcessor, BlipForQuestionAnswering

# -------- 必填路径 ------------
BASE   = "/home/guest_a/yuezhao/dongwenou/NLP_Final/blip-vqa-base"
CKPT   = "./blip_vqa_final3.pt"                 # 微调权重
ROOT   = "/home/guest_a/yuezhao/dongwenou/NLP_Final"
VAL_IMG   = f"{ROOT}/val2014"
VAL_QPATH = f"{ROOT}/v2_OpenEnded_mscoco_val2014_questions.json"
VAL_APATH = f"{ROOT}/v2_mscoco_val2014_annotations.json"
PREFIX = "COCO_val2014"

SAVE_DIR = "attention_vis"
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# -------- 数据集构建 -----------
def load_json(path, key):
    with open(path) as f: return json.load(f)[key]

def id2name(img_id): return f"{PREFIX}_{img_id:012d}.jpg"

def build_val_list():
    qs   = load_json(VAL_QPATH, "questions")
    anns = {a["question_id"]: a for a in load_json(VAL_APATH, "annotations")}
    lst  = []
    for q in qs:
        a = anns.get(q["question_id"])
        if not a: continue
        img = os.path.join(VAL_IMG, id2name(q["image_id"]))
        lst.append((img, q["question"]))
    return lst

# -------- 可视化函数 -----------
def overlay_heatmap(img_pil, heat, cmap="jet", alpha=0.45):
    img = np.array(img_pil.convert("RGB"))
    heat = cv2.resize(heat, (img.shape[1], img.shape[0]))
    heat = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heat_color, alpha, img, 1 - alpha, 0)
    return overlay[..., ::-1]

# ---------------------- main ----------------------
def main(num, indices):
    val_list = build_val_list()
    chosen = [val_list[i] for i in indices] if indices else random.sample(val_list, num)

    proc = BlipProcessor.from_pretrained(BASE)
    model = BlipForQuestionAnswering.from_pretrained(BASE)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)

    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.config.return_dict = True

    model.to(DEVICE).eval()

    for i, (img_path, ques) in enumerate(tqdm(chosen, desc="Attention Vis")):
        img = Image.open(img_path).convert("RGB")
        inputs = proc(img, ques, return_tensors="pt").to(DEVICE)

        # 首先生成答案序列
        generated = model.generate(**inputs, max_length=10, num_beams=5)
        answer = proc.tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\n图像文件: {img_path}")
        print(f"问题: {ques}\n答案: {answer}")

        # 明确传入decoder_input_ids并捕获decoder的cross attention
        decoder_input_ids = generated[:, :-1]  # 去掉最后一个token以作为输入
        with torch.no_grad():
            encoder_out = model.vision_model(inputs['pixel_values'])
            outputs = model.text_decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_out.last_hidden_state,
                output_attentions=True,
                return_dict=True
            )

        # 获取cross attentions
        cross_attn = outputs.cross_attentions  # decoder对视觉特征的cross-attention
        if cross_attn is None:
            print("没有获得 cross attention，检查模型配置！")
            continue

        # 取最后一层平均所有head的attention，并且选择第一个生成token对图像的attention
        attn = cross_attn[-1][0].mean(0)

        # 跳过CLS token
        heat = attn[0, 1:]  # 跳过CLS
        side = int(math.sqrt(heat.size(-1)))  # 应该为24
        heat = heat.reshape(side, side).cpu().numpy()
        heat = (heat - heat.min()) / (heat.ptp() + 1e-8)

        overlay = overlay_heatmap(img, heat)
        save_path = os.path.join(SAVE_DIR, f"vis_{i:02d}.png")
        Image.fromarray(overlay).save(save_path)

        plt.imshow(overlay); plt.axis("off")
        plt.title(f"Q: {ques}\nA: {answer}")
        plt.show()

        print(f"已保存可视化: {save_path}")
        print()

# ---------------------- 执行入口 ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=3, help="随机可视化样本数")
    parser.add_argument("--indices", nargs="*", type=int, help="指定 val 集索引")
    parser.add_argument("--seq", type=int, help="依次可视化前 seq 张")

    args = parser.parse_args()

    if args.seq:
        idxs = list(range(args.seq))
        main(args.seq, idxs)
    else:
        idxs = args.indices if args.indices else None
        main(args.num, idxs)