import os
import sys
from pathlib import Path
base = Path(__file__).resolve().parent
sys.path.append(base.parent.parent.as_posix())

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
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from model.Character_classifier.AnimeHeadDetector.AnimeHeadDetector import AnimeHeadDetector
from types import SimpleNamespace
# dataset_path = r"G:\hoc\private\Anime\data\Character_Classifier_data\processed\dataset_balanced\train"  # <-- change this to your dataset root

# class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

# output_file = "Anime_character_classes.txt"
# with open(output_file, "w", encoding="utf-8") as f:
#    for name in class_names:
#        f.write(name + "\n")

# print(f"✅ Extracted {len(class_names)} class names and saved to {output_file}")

base = Path(__file__).resolve().parent

def arg_parse(image_path, output_dir):
    """
    Return arguments as an object instead of using argparse
    """
    cfgfile_path = base.parent.parent / "model" / "Character_classifier" / "AnimeHeadDetector" / "head.cfg"
    weightsfile_path = base.parent.parent / "model" / "Character_classifier" / "AnimeHeadDetector" / "head.weights"
    return SimpleNamespace(
        images=image_path,
        det=output_dir,
        cfgfile= cfgfile_path,
        weightsfile= weightsfile_path
    )


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

def predict_with_image_path(img_path,model = None):

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    try:
        img = Image.open(img_path).convert("RGB")
    except (UnidentifiedImageError, FileNotFoundError):
        print(f" Could not open image: {img_path}")
        return None

    img_tensor = transform(img).unsqueeze(0)
    class_path = base.parent.parent / "src" / "classifier" / "Anime_character_classes.txt"
    with open(class_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    with torch.no_grad():
        outputs = model(img_tensor)
        pred_class = torch.argmax(outputs, dim=1).item()
        print(f"✅ Prediction for '{os.path.basename(img_path)}' → Class Index: {pred_class} -> Class Name: {class_names[pred_class]}")
    return pred_class

def predict_with_image(image_path = None, model = None):

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    class_name_path = base.parent.parent / "src" / "classifier" / "Anime_character_classes.txt"
    with open(class_name_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    output_dir = r"G:\hoc\private\Anime\test"
    args = arg_parse(image_path, output_dir)
    detector = AnimeHeadDetector(args.cfgfile, args.weightsfile)
    images = args.images

    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
    if not os.path.exists(args.det):
        os.makedirs(args.det)
        
    for imfn in tqdm(imlist):
        im = cv2.imread(imfn)
        # im2 = detector.detectAndVisualize(im) # pillow image
        # cv2.imwrite(osp.join(args.det, osp.basename(imfn)), im2)
        h, w = im.shape[:2]
        im3, results = detector.detectAndCrop(im)
        if im3: 
            for i in range(0, len(im3)): # ensure at least one detection exists
                crop = im3[i]

                # convert BGR -> RGB
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                # convert NumPy -> PIL image
                pil_img = Image.fromarray(crop_rgb)

                # apply preprocessing transform
                img_tensor = transform(pil_img).unsqueeze(0)  # [1, 3, 224, 224]

                # inference
                with torch.no_grad():
                    outputs = model(img_tensor)
                    pred_class = torch.argmax(outputs, dim=1).item()

                cv2.rectangle(im, (results[i]['l'], results[i]['t']), (results[i]['r'], results[i]['b']), (0, 255, 0) , 5)
                cv2.imwrite(osp.join(args.det, osp.basename(imfn)), im)

                text = class_names[pred_class]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = max(0.4, min(1.2, h / 800))   # smaller overall scale
                thickness = int(max(1, h / 400))   

                color = (0, 255, 0)
                text_x = results[i]['l']      # or whatever your left x coordinate key is
                text_y = results[i]['b'] + 30 # move text 30 pixels below bottom

                cv2.putText(im, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
                cv2.putText(im, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                cv2.imwrite(osp.join(args.det, osp.basename(imfn)), im)
                print(f"✅ Prediction for '{os.path.basename(imfn)}' → Class Index: {pred_class} -> Class Name: {class_names[pred_class]}")
            else:
                print(f"⚠️ No detection found in '{os.path.basename(imfn)}'")
        

        # Sample usage of detectAndCrop
        # imgs = detector.detectAndCrop(im)
        # for i, img in enumerate(imgs):
        #     cv2.imwrite(osp.join(args.det, '{}_{}.jpg'.format(osp.basename(imfn), i)), img)
        # results = detector.detect(im)
        # for i, result in enumerate(results):
        #     width = result['r'] - result['l'] + 1
        #     height = result['b'] - result['t'] + 1
        #     area = width * height
        #     print(f"Detection {i}: area = {area} pixels²")
        
        
        



    torch.cuda.empty_cache()

   

    


def Vit_Model():
    num_classes = 173
    model_path = base.parent.parent / "model" / "Character_classifier" / "vit_model.pth"
    model_Vision_Transformer = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model_Vision_Transformer.heads.head = nn.Linear(model_Vision_Transformer.heads.head.in_features, num_classes)
    model_Vision_Transformer.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_Vision_Transformer.eval()

    print(" Model ViT loaded successfully and ready for inference.")
    return model_Vision_Transformer

def Resnet50_Model():
    num_classes = 173
    model_path = base.parent.parent / "model" / "Character_classifier" / "resnet50_finetuned.pth"
    model_resnet50 = models.resnet50(pretrained=True)
    model_resnet50.fc = nn.Linear(model_resnet50.fc.in_features, num_classes)
    model_resnet50.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_resnet50.eval()

    print(" Model Resnet50 loaded successfully and ready for inference.")
    return model_resnet50

def EfficientNetB0_Model():
    num_classes = 173
    model_path = base.parent.parent / "model" / "Character_classifier" / "EfficientnetB0_finetuned.pth"
    model_efficientnet_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model_efficientnet_b0.classifier[1] = nn.Linear(model_efficientnet_b0.classifier[1].in_features, num_classes)
    model_efficientnet_b0.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_efficientnet_b0.eval()

    print(" Model EfficientNetB0 loaded successfully and ready for inference.")
    return model_efficientnet_b0


if __name__ == "__main__":
    Vit_model = Vit_Model()
    Vit_model.eval()
    image_test1_path = base.parent.parent / "test" / "New folder" /"images.webp"
    image_test2_path = base.parent.parent / "test" / "New folder" /"KallenGRPortraitZoom.webp"
    image_test3_path = base.parent.parent / "test" / "New folder" /"lcsdal9y89y21.webp"
    image_test4_path = base.parent.parent / "test" / "New folder" /"d616356e8da246be39095dadd1e891d7.jpg"
    predict_with_image_path(image_test1_path, model = Vit_model)
    predict_with_image_path(image_test2_path, model = Vit_model)
    predict_with_image_path(image_test3_path, model = Vit_model)
    predict_with_image_path(image_test4_path, model = Vit_model)
    ####################################################
    Resnet50_model = Resnet50_Model()
    predict_with_image_path(image_test1_path, model = Resnet50_model)
    predict_with_image_path(image_test2_path, model = Resnet50_model)
    predict_with_image_path(image_test3_path, model = Resnet50_model)
    predict_with_image_path(image_test4_path, model = Resnet50_model)
    ####################################################
    EfficientNetB0_model = EfficientNetB0_Model()
    predict_with_image_path(image_test1_path, model = EfficientNetB0_model)
    predict_with_image_path(image_test2_path, model = EfficientNetB0_model)
    predict_with_image_path(image_test3_path, model = EfficientNetB0_model)
    predict_with_image_path(image_test4_path, model = EfficientNetB0_model)  
    #####################################################
    
    predict_with_image(image_test4_path, model = Vit_model)
    