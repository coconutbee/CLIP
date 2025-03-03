import os
import shutil
import torch
import open_clip
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

#model type: 'RN50', 'RN50-quickgelu', 'RN101', 'RN101-quickgelu', 'RN50x4', 'RN50x16', 'RN50x64', 
#model type: 'ViT-B-32', 'ViT-B-32-quickgelu', 'ViT-B-16', 'ViT-L-14', 'ViT-L-14-336'
# 載入 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, _ = open_clip.create_model_and_transforms("ViT-L-14-336", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-L-14-336")
model = model.to(device)

# 定義圖片分類標籤
categories = {
    # "poster": "A movie or advertisement poster",
    # "face_near_edge": "A photo where the person's face is too close to the edge",
    # "small_face": "A photo where the face is very small compared to the whole image",
    "non_person": "An image that contains no human faces, such as landscapes, objects, or animals",
    "portrait_painting": "A painting of a person",  
    # "partial_face": "A photo of a person where only part of the face is visible, such as only the mouth, nose, or side profile",
    "landscape": "A landscape photo without any person",
    "full_face": "A clear and complete human face with both eyes, nose, and mouth fully visible"
}

# 設定資料夾
image_folder = "cfpfp_clean"   # 未分類圖片的資料夾
sorted_folder = "cfpfp_sorted"      # 分類後的資料夾
os.makedirs(sorted_folder, exist_ok=True)

# 建立分類資料夾
for category in categories.keys():
    os.makedirs(os.path.join(sorted_folder, category), exist_ok=True)
os.makedirs(os.path.join(sorted_folder, "uncertain"), exist_ok=True)  # 存放不確定的圖片

# 讀取所有 .jpg 圖片
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# 轉換標籤為 CLIP 編碼
text_inputs = tokenizer(list(categories.values())).to(device)

print("Processing images with CLIP...")

for image_file in tqdm(image_files, desc="Classifying images"):
    image_path = os.path.join(image_folder, image_file)

    # 讀取並預處理圖片
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    # 計算 CLIP 相似度
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    # 找到最匹配的類別
    best_match_idx = similarity.argmax().item()
    best_category = list(categories.keys())[best_match_idx]
    confidence_score = similarity[0, best_match_idx].item()

    # 設定信心值閾值（低於 0.34 則視為不確定）
    if confidence_score < 0.25:
        best_category = "uncertain"

    # 移動圖片到對應分類資料夾
    target_path = os.path.join(sorted_folder, best_category, image_file)
    shutil.move(image_path, target_path)
    print(f"Moved {image_file} → {best_category} (Confidence: {confidence_score:.2f})")

print("Classification completed!")
