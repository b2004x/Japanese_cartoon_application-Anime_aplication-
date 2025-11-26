import sys
import os
import pandas as pd
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
BASE_DIR = os.path.dirname(__file__)
module_path = os.path.join(BASE_DIR, "private", "Anime")
if module_path not in sys.path:
    sys.path.append(module_path)
from recomender.dataset import load_anime_data, preprocess_anime_data, parse_duration, preprocess_anime_data_sypnosis
from recomender.embeddings_similarity.embeddings_similarity import hybrid_recommend_by_each, embedding_sypnosis, recommend_by_name_fuzzy, recommend_by_names, hybrid_recommend_vectorized
from recomender.metadata_filtering.metadata_filtering import metadata_similarity_hybrid, metadata_similarity,recommend_for_each_favorite
from utils.utils import minmax_normalize, load_embeddings, find_closest_title
import numpy as np
import faiss
from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import math
from torchvision import datasets, transforms
from PIL import Image, UnidentifiedImageError
from model.Character_classifier.AnimeHeadDetector.AnimeHeadDetector import AnimeHeadDetector
import cv2
import argparse
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os.path as osp
from tqdm import tqdm
import base64
from types import SimpleNamespace
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
from pathlib import Path
print(os.getcwd())
def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(i) for i in obj]
    elif isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        else:
            return None
    else:
        return obj

def normalize_name(name):
    return name.lower().strip()
    

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for safety
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Preload data once at startup
# ----------------------------
base = Path(__file__).resolve().parent
df_anime_path =  base.parent / "data" / "Recomended_Anime_data" / "raw" / "MyAnimeList-Database-master" / "data" / "anime.csv"
df_anime_sypnosis_path = base.parent / "data" / "Recomended_Anime_data" / "raw" / "MyAnimeList-Database-master" / "data" / "anime_with_synopsis.csv"
embedding_path = base / "recomender" / "embeddings_similarity" / "anime_embeddings.npy"

print("Loading datasets...")
df_anime = pd.read_csv(df_anime_path)
df_anime = preprocess_anime_data(df_anime)

df_anime_sypnosis = pd.read_csv(df_anime_sypnosis_path)
df_anime_sypnosis = preprocess_anime_data_sypnosis(df_anime_sypnosis)

Name_to_idx = {Name: i for i, Name in enumerate(df_anime_sypnosis["Name"])}

embeddings = np.load(embedding_path).astype("float32")


index_path = base / "recomender" / "embeddings_similarity" / "anime_index.faiss"
print(index_path.exists())
print(index_path.resolve())
index = faiss.read_index(
   str(index_path)
)

print("Data loaded. Ready to accept requests.")

# ----------------------------
# Optional cache to speed up repeated requests
# ----------------------------
recommendation_cache = {}


@app.get("/recommend/")

async def get_recommendations(names: str = Query(..., description="Comma-separated anime names")):
    print("Received recommendation request")

    if not names:
        return JSONResponse({"error": "names parameter required"}, status_code=400)

    # Normalize input
    user_likes = [name.strip() for name in names.split(",") if name.strip()]
    cache_key = ",".join(user_likes)

    # Return cached result if available
    if cache_key in recommendation_cache:
        print("Cache hit")
        return JSONResponse({"recommendations": recommendation_cache[cache_key]})

    print(f"Generating recommendations for: {user_likes}")

    # Call your recommender
    sections = hybrid_recommend_by_each(
        df_anime=df_anime,
        df_anime_sypnosis=df_anime_sypnosis,
        Names=user_likes,
        k=10,  # smaller k = faster
        alpha=0.5,
        mode="hybrid",
        embeddings=embeddings,
        index=index,
        Name_to_idx=Name_to_idx,
    )

    # Convert DataFrames to JSON-friendly format
    response = {}
    for liked, df_recs in sections.items():
        print(f"\nBecause you liked {liked}:\n", df_recs.head())
        response[liked] = df_recs[
            ["Name", "Score", "Genres", "hybrid_score", "explanation"]
        ].to_dict(orient="records")

    # Cache the result
    recommendation_cache[cache_key] = response

    response = clean_json(response)

    return JSONResponse({"recommendations": response})


#############################################################################################

# def arg_parse(image_path, output_dir):
#     """
#     Parse arguements to the detect module
#     """
    
#     parser = argparse.ArgumentParser(description='Anime head detector based on Yolo V3.')
   
#     parser.add_argument("--images", dest = 'images', help = 
#                         "Image / Directory containing images to perform detection upon",  
#                         default = image_path, type = str)
#     parser.add_argument("--det", dest = 'det', help =  
#                         "Image / Directory to store detections to",    
#                         default = output_dir, type = str)
#     parser.add_argument("--cfg", dest = 'cfgfile', help = 
#                         "Config file",
#                         default = r"G:\hoc\private\Anime\model\Character_classifier\AnimeHeadDetector\head.cfg", type = str)
#     parser.add_argument("--weights", dest = 'weightsfile', help = 
#                         "weightsfile",
#                         default = r"G:\hoc\private\Anime\model\Character_classifier\AnimeHeadDetector\head.weights", type = str)
    
#     return parser.parse_args()

def arg_parse(image_path, output_dir):
    """
    Return arguments as an object instead of using argparse
    """
    cfgfile_path = base.parent / "model" / "Character_classifier" / "AnimeHeadDetector" / "head.cfg"
    weightsfile_path = base.parent / "model" / "Character_classifier" / "AnimeHeadDetector" / "head.weights"
    return SimpleNamespace(
        images=image_path,
        det=output_dir,
        cfgfile= cfgfile_path,
        weightsfile= weightsfile_path
    )


def Vit_Model():
    num_classes = 173
    model_path = base.parent / "model" / "Character_classifier" / "vit_model.pth"
    model_Vision_Transformer = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model_Vision_Transformer.heads.head = nn.Linear(model_Vision_Transformer.heads.head.in_features, num_classes)
    model_Vision_Transformer.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_Vision_Transformer.eval()

    print(" Model ViT loaded successfully and ready for inference.")
    return model_Vision_Transformer

def Resnet50_Model():
    num_classes = 173
    model_path = base.parent / "model" / "Character_classifier" / "resnet50_finetuned.pth"
    model_resnet50 = models.resnet50(pretrained=True)
    model_resnet50.fc = nn.Linear(model_resnet50.fc.in_features, num_classes)
    model_resnet50.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_resnet50.eval()

    print(" Model Resnet50 loaded successfully and ready for inference.")
    return model_resnet50

def EfficientNetB0_Model():
    num_classes = 173
    model_path = base.parent / "model" / "Character_classifier" / "EfficientnetB0_finetuned.pth"
    model_efficientnet_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model_efficientnet_b0.classifier[1] = nn.Linear(model_efficientnet_b0.classifier[1].in_features, num_classes)
    model_efficientnet_b0.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_efficientnet_b0.eval()

    print(" Model EfficientNetB0 loaded successfully and ready for inference.")
    return model_efficientnet_b0



    

@app.post("/classify")


async def recognize_character(file: UploadFile = File(...)):
    print(" Received character classification request")
    # 📂 Save uploaded file
    input_dir = "uploads"
    os.makedirs(input_dir, exist_ok=True)
    image_path = osp.join(input_dir, file.filename)

    with open(image_path, "wb") as f:
        f.write(await file.read())

    # 🧠 Load model and preprocessing
    Vit_model = Vit_Model()
    Vit_model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # 📜 Load class labels
    label_path = base /  "Anime_character_classes.txt"
    with open(label_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # 🕵️ Initialize head detector
    output_dir = Path(r"test")
    args = arg_parse(image_path, output_dir)
    detector = AnimeHeadDetector(args.cfgfile, args.weightsfile)
    images = args.images

    try:
        imlist = [osp.join(osp.realpath('.'), images, img)
                  for img in os.listdir(images)
                  if os.path.splitext(img)[1].lower() in ['.png', '.jpeg', '.jpg']]
    except NotADirectoryError:
        imlist = [osp.join(osp.realpath('.'), images)]

    if not os.path.exists(args.det):
        os.makedirs(args.det)

    predictions = []

    for imfn in tqdm(imlist):
        im = cv2.imread(imfn)
        h, w = im.shape[:2]
        im3, results = detector.detectAndCrop(im)

        if not im3:
            print(f"⚠️ No detection in {imfn}")
            continue

        for i, crop in enumerate(im3):
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            img_tensor = transform(pil_img).unsqueeze(0)

            with torch.no_grad():
                outputs = Vit_model(img_tensor)
                pred_class = torch.argmax(outputs, dim=1).item()
                confidence = torch.softmax(outputs, dim=1)[0, pred_class].item()

            class_name = class_names[pred_class]
            predictions.append({
                "character_name": class_name,
                "confidence": confidence
            })

            # 🟩 Draw bounding box and label
            color = (0, 255, 0)
            thickness = 3
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.4, min(1.2, h / 800))

            l, t, r, b = results[i]['l'], results[i]['t'], results[i]['r'], results[i]['b']
            cv2.rectangle(im, (l, t), (r, b), color, thickness)
            text = f"{class_name} ({confidence:.2f})"

            # put label above box
            text_y = max(t - 10, 20)
            cv2.putText(im, text, (l, text_y), font, font_scale, (0, 0, 0), 3)
            cv2.putText(im, text, (l, text_y), font, font_scale, (255, 255, 255), 1)

        # 🖼️ Convert the last annotated image to base64
        _, buffer = cv2.imencode('.jpg', im)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

    if not predictions:
        return {"error": "No character detected"}

    best = max(predictions, key=lambda x: x["confidence"])

    # 🧾 Return both prediction and annotated image
    return {
        "best_prediction": best,
        "all_predictions": predictions,
        "annotated_image": img_base64
    }

#################################################################################################

device = torch.device('cpu')
_real_load_state_dict_from_url = torch.hub.load_state_dict_from_url


def cpu_load_state_dict_from_url(url, *args, **kwargs):
    kwargs["map_location"] = torch.device("cpu")  # force CPU
    return _real_load_state_dict_from_url(url, *args, **kwargs)

index_to_tag = {}
index_to_tag_path = base.parent / "model" / "Captions_and_Tagging" / "Tagging model" / "Index_to_Tags.txt"
with open(index_to_tag_path, "r", encoding="utf-8") as f:
    for line in f:
        idx, tag = line.strip().split(maxsplit=1)
        index_to_tag[int(idx)] = tag


def tagging_model_output(outputs, threshold_path):
    """Convert model output logits into tags above threshold"""
    best_thresholds = np.load(threshold_path)
    thresholds = torch.tensor(best_thresholds)

    probs = torch.sigmoid(outputs)[0]
    high_conf_idxs = (probs > thresholds).nonzero(as_tuple=True)[0]

    tags = []
    for idx in high_conf_idxs:
        tag = index_to_tag.get(idx.item(), f"unknown_{idx.item()}")
        tags.append(tag)

    return tags


def generate_caption(image_path):
    """Run BLIP caption model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blip_path = base.parent / "model" / "Captions_and_Tagging" / "Model_Second_half" / "blip-anime-half2-final"
    model_path = base.parent / "model" / "Captions_and_Tagging" / "Model_Second_half" / "blip-anime" / "checkpoint-80000"
    processor = BlipProcessor.from_pretrained(
        blip_path
    )
    model = BlipForConditionalGeneration.from_pretrained(
        model_path
    ).to(device)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


@app.post("/caption_tag/")
async def caption_and_tag(file: UploadFile = File(...)):
    print("Starting caption and tagging process...")
    # Save uploaded image in memory
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tags = 500

    # Load tagging model
    # torch.hub.load_state_dict_from_url = lambda url, *a, **k: torch.hub.load_state_dict_from_url(
    #     url, map_location=device
    # )
    torch.hub.load_state_dict_from_url = cpu_load_state_dict_from_url
    model = torch.hub.load("RF5/danbooru-pretrained", "resnet50", trust_repo=True)
    model[1][8] = nn.Linear(in_features=512, out_features=num_tags)
    model.to(device)
    model_tagging_path = base.parent / "model" / "Captions_and_Tagging" / "Tagging model" / "model_danboruu_resnet50_finetuned.pth"
    model.load_state_dict(
        torch.load(
            model_tagging_path,
            map_location=device,
        )
    )
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)

    tags_threshold_path = base.parent / "model" / "Captions_and_Tagging" / "Tagging model" / "best_thresholds_Danbooru_Resnet50.npy"
    tags = tagging_model_output(
        outputs,
        tags_threshold_path,
    )
    
    # Save temp image for captioning
    temp_path = "temp_upload.jpg"
    image.save(temp_path)
    caption = generate_caption(temp_path)
    tags = [str(t) for t in tags]
    print("TAGS PYTHON VALUE =", tags)
    print("TAGS PYTHON TYPES =", [type(t) for t in tags])
    return {"caption": caption, "tags": tags}