from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from functools import partial
import numpy as np
from torchvision import models
from pathlib import Path

device = torch.device('cpu')
_real_load_state_dict_from_url = torch.hub.load_state_dict_from_url
base = Path(__file__).resolve().parent

index_to_tag = {}
index_to_tag_path = base.parent.parent / "model" / "Captions_and_Tagging" / "Tagging model" / "Index_to_Tags.txt"

with open(index_to_tag_path, "r", encoding="utf-8") as f:
    for line in f:
        idx, tag = line.strip().split(maxsplit=1)
        index_to_tag[int(idx)] = tag


def cpu_load_state_dict_from_url(url, *args, **kwargs):
    kwargs['map_location'] = device
    return _real_load_state_dict_from_url(url, *args, **kwargs)




def tagging_model_output(outputs, threshold_path):
    best_thresholds  = np.load(threshold_path)
    thresholds = torch.tensor(best_thresholds) 

    probs = torch.sigmoid(outputs)[0]
    high_conf_idxs = (probs > thresholds).nonzero(as_tuple=True)[0]

    for idx in high_conf_idxs:
        tag = index_to_tag.get(idx.item(), f"unknown_{idx.item()}")
        print(f"{tag}: {probs[idx].item():.4f}")





def Caption(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor_path = base.parent.parent / "model" / "Captions_and_Tagging" / "Model_Second_half" / "blip-anime-half2-final"
    model_blip_path = base.parent.parent / "model" / "Captions_and_Tagging" / "Model_Second_half" / "blip-anime" / "checkpoint-80000"
    processor = BlipProcessor.from_pretrained(processor_path)
    model = BlipForConditionalGeneration.from_pretrained(model_blip_path)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(caption)


def Tagging_Danboruu_Resnet50(image_path):

    # Temporarily patch torch.load inside torch.hub
    # torch.hub.load_state_dict_from_url = lambda *a, **k: cpu_load(*a, **k)
    torch.hub.load_state_dict_from_url = cpu_load_state_dict_from_url
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_tags = 500
    model = torch.hub.load('RF5/danbooru-pretrained','resnet50', trust_repo=True, force_reload=True)
    model[1][8] = nn.Linear(in_features=512, out_features=num_tags)
    model.to(device)
    print("Model loaded successfully.")
    model_path = base.parent.parent / "model" / "Captions_and_Tagging" / "Tagging model" / "model_danboruu_resnet50_finetuned.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    transform = transforms.Compose([
    transforms.Resize((224, 224)),           
    transforms.ToTensor(),                   
    transforms.Normalize(                    
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
    ])

    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  

   
    with torch.no_grad():
        outputs = model(input_tensor)  

    
    # topk = 10
    # probs = torch.sigmoid(outputs)[0]  
    # top_probs, top_idxs = probs.topk(topk)

    # print("Top predicted tags and probabilities:")
    # for i in range(topk):
    #     print(f"Tag {top_idxs[i].item()}: {top_probs[i].item():.4f}")
    tagging_threshold_path = base.parent.parent / "model" / "Captions_and_Tagging" / "Tagging model" / "best_thresholds_Danbooru_Resnet50.npy"
    tagging_model_output(outputs, tagging_threshold_path)



def Tagging_Resnet50(image_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 500
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    model_path = base.parent.parent / "model" / "Captions_and_Tagging" / "Tagging model" / "model_resnet50_finetuned.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
    transforms.Resize((224, 224)),           
    transforms.ToTensor(),                   
    transforms.Normalize(                    
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
    ])

    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  

   
    with torch.no_grad():
        outputs = model(input_tensor)  

    tagging_threshold = base.parent.parent / "model" / "Captions_and_Tagging" / "Tagging model" / "best_thresholds_resnet50.npy"
    
    tagging_model_output(outputs, tagging_threshold)





def Tagging_Autotagger_Resnet152(image_path):
    num_classes = 500
    pretrained_weights_path = base.parent.parent / "model" / "Captions_and_Tagging" / "Tagging model" / "model_resnet152_finetuned_second_half.pth"

    model = models.resnet152(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    state_dict = torch.load(pretrained_weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    model.eval()

    transform = transforms.Compose([
    transforms.Resize((224, 224)),           
    transforms.ToTensor(),                   
    transforms.Normalize(                    
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
    ])

    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  

   
    with torch.no_grad():
        outputs = model(input_tensor)  
    tagging_model_threshold_path = base.parent.parent / "model" / "Captions_and_Tagging" / "Tagging model" / "best_thresholds_resnet152_autotagger.npy"
    tagging_model_output(outputs, tagging_model_threshold_path)
    

if __name__ == "__main__":
    image_path = base.parent.parent / "test" / "New folder" /"anime-k-on_00178876.jpg"

    
    print("Danboruu Resnet 50 Tagging Results:")
    Tagging_Danboruu_Resnet50(image_path)
    print("\nResnet 50 Tagging Results:")
    Tagging_Resnet50(image_path)
    print("\nAutotagger Resnet 152 Tagging Results:")
    Tagging_Autotagger_Resnet152(image_path)
    Caption(image_path)
    

