# An anime application for providing recommendations, classifying characters, tagging illustrations, and generating captions

## Description
- **An anime application for providing recommendations, classifying characters, tagging illustrations, and generating captions. This is a personal project that I am very passionate about**

## This project divede into Three parts:

### Anime Recommendation system:
- **This system automatically recommends anime similar to your favorite titles by leveraging word embeddings and metadata filtering. This allows you to always find new and fresh anime titles that suit your preferences**

### Character classification:
- **This module predict multiple charactes in an anime images using deep learning model and machine learning techniques. It analyzes the visual features and metadata to classify the character accurately based on an existing database**

### illustrations tagging and generating caption:
- **This system automatically tags illustrations with relevant keywords and generates descriptive captions using deep learning model. Helping artists save time, effort and get their work discovered more easily.**

## Features
- **Find and save user favourite anime in database**
- **Show anime details**
- **Recommend anime based on user favourite anime**
- **Detect multiple human faces in anime images, drawing bounding box around the face and predict the character**
- **Automatically tag illustrations with relevant keywords.**
- **Generate descriptive captions for anime images**

## Sample
### Anime search by genre and name
![alt text](test/test2/Anime%20search%20by%20name%20and%20genre.PNG)

### Anime list
![alt text](test/test2/Anime%20list.PNG)

### Anime details
![alt text](test/test2/Anime%20details.PNG)

### Recommend anime
![alt text](test/test2/Recommend%20anime.PNG)

### Detect human faces in anime images and predict the characters
![alt text](test/test2/Anime%20Characters%20classification.jfif)

### Automatically tag illustrations and Generate descriptive captions 
![alt text](test/test2/Caption%20and%20tagging.PNG)


## Dataset
### Anime lists and details 
- **dataset: https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020?select=anime_with_synopsis.csv**


### Anime human face detection with characters classification
- **dataset: https://www.kaggle.com/datasets/nguyengiabach1810/anime-character-classification-2**
- **https://www.kaggle.com/datasets/lngthnhan/anime-character-classes**

```

moeimouto-faces/
├── 000_hatsune_miku/
│   ├── face_128_326_108.png
│   ├── face_136_245_126.png
│   ├── face_165_132_79.png
│   ├── face_168_638_76.png
│   ├── face_208_170_115.png
│   ├── face_209_55_78.png
│   ├── face_214_99_108.png
│   ├── face_227_303_150.png
│   ├── face_250_136_15.png
│   ├── face_250_181_154.png
│   ├── face_260_173_131.png
│   ├── face_265_170_51.png
│   ├── face_277_225_56.png
│   ├── face_27_136_113.png
│   ├── face_286_77_58.png
│   ├── face_295_113_81.png
│   ├── face_303_158_140.png
│   ├── face_311_221_73.png
│   ├── face_322_170_135.png
│   ├── face_322_322_38.png
│   ├── face_326_130_21.png
│   ├── face_328_209_46.png
│   ├── face_332_124_33.png
│   ├── face_341_160_137.png
│   ├── face_357_256_119.png
│   ├── face_361_115_129.png
│   ├── face_363_253_144.png
│   ├── face_370_580_110.png
│   ├── face_385_110_90.png
│   ├── color.csv
│   └── ... 35 more images
│
├── 001_kinomoto_sakura/
├── 002_suzumiya_haruhi/
├── 003_fate_testarossa/
├── 004_takamachi_nanoha/
├── 005_lelouch_lamperouge/
├── 006_akiyama_mio/
├── 007_nagato_yuki/
├── 008_shana/
├── 009_hakurei_reimu/
├── 010_izumi_konata/
├── 011_kirisame_marisa/
├── 012_asahina_mikuru/
├── 013_saber/
├── 014_hiiragi_kagami/
├── 015_c.c/
├── 016_furukawa_nagisa/
├── 017_louise/
├── 018_kagamine_rin/
├── 019_ayanami_rei/
├── 020_remilia_scarlet/
├── 021_hirasawa_yui/
├── 022_kururugi_suzaku/
├── 023_hiiragi_tsukasa/
├── 024_fujibayashi_kyou/
├── 025_souryuu_asuka_langley/
├── 026_tohsaka_rin/
├── 027_izayoi_sakuya/
├── 028_tainaka_ritsu/
├── 029_kallen_stadtfeld/
└── ... 143 more character folders

```
### Tagging dataset 
- **dataset: https://www.kaggle.com/datasets/mylesoneill/tagged-anime-illustrations**
- **https://www.kaggle.com/code/b2004x/anime-tagging-data-classes/output?scriptVersionId=267078021**

```
data/
└── danbooru/
    ├── danbooru-images/
    │   ├── 0000/
    │   ├── 0001/
    │   ├── 0002/
    │   ├── 0003/
    │   ├── 0004/
    │   ├── 0005/
    │   ├── 0006/
    │   ├── 0007/
    │   ├── 0008/
    │   ├── 0009/
    │   ├── 0010/
    │   ├── 0011/
    │   ├── 0012/
    │   ├── 0013/
    │   ├── 0014/
    │   ├── 0015/
    │   ├── 0016/
    │   ├── 0017/
    │   ├── 0018/
    │   ├── 0019/
    │   ├── 0020/
    │   ├── 0021/
    │   ├── 0022/
    │   ├── 0023/
    │   ├── 0024/
    │   ├── 0025/
    │   ├── 0026/
    │   ├── 0027/
    │   ├── 0028/
    │   ├── 0029/
    │   └── ... 121 more folders
    │
    └── danbooru-metadata/
        ├── 201700.json
        ├── 201701.json
        ├── 201702.json
        ├── 201703.json
        ├── 201704.json
        ├── 201705.json
        ├── 201706.json
        ├── 201707.json
        ├── 201708.json
        ├── 201709.json
        ├── 201710.json
        ├── 201711.json
        ├── 201712.json
        ├── 201713.json
        ├── 201714.json
        ├── 201715.json
        └── 201716.json
```
-**json contain the label of the image**

### Caption dataset
- **dataset: https://huggingface.co/datasets/none-yet/anime-captions/viewer/default/train?p=2&views%5B%5D=train**

```
anime_captions/
├── train/                         
│   ├── shards/                    # all HuggingFace .arrow files for training
│   │   ├── anime-captions-train-00000-of-00057.arrow
│   │   ├── anime-captions-train-00001-of-00057.arrow
│   │   ├── anime-captions-train-00002-of-00057.arrow
│   │   ├── anime-captions-train-00003-of-00057.arrow
│   │   ├── anime-captions-train-00004-of-00057.arrow
│   │   ├── anime-captions-train-00005-of-00057.arrow
│   │   ├── anime-captions-train-00006-of-00057.arrow
│   │   ├── anime-captions-train-00007-of-00057.arrow
│   │   ├── anime-captions-train-00008-of-00057.arrow
│   │   ├── anime-captions-train-00009-of-00057.arrow
│   │   ├── anime-captions-train-00010-of-00057.arrow
│   │   ├── anime-captions-train-00011-of-00057.arrow
│   │   ├── anime-captions-train-00012-of-00057.arrow
│   │   ├── anime-captions-train-00013-of-00057.arrow
│   │   ├── anime-captions-train-00014-of-00057.arrow
│   │   ├── anime-captions-train-00015-of-00057.arrow
│   │   ├── anime-captions-train-00016-of-00057.arrow
│   │   ├── anime-captions-train-00017-of-00057.arrow
│   │   ├── anime-captions-train-00018-of-00057.arrow
│   │   ├── anime-captions-train-00019-of-00057.arrow
│   │   ├── anime-captions-train-00020-of-00057.arrow
│   │   ├── anime-captions-train-00021-of-00057.arrow
│   │   ├── anime-captions-train-00022-of-00057.arrow
│   │   ├── anime-captions-train-00023-of-00057.arrow
│   │   ├── anime-captions-train-00024-of-00057.arrow
│   │   ├── anime-captions-train-00025-of-00057.arrow
│   │   ├── anime-captions-train-00026-of-00057.arrow
│   │   ├── anime-captions-train-00027-of-00057.arrow
│   │   ├── anime-captions-train-00028-of-00057.arrow
│   │   ├── anime-captions-train-00029-of-00057.arrow
│   │   └── ... (remaining shards up to 00056-of-00057)
│   │
│   ├── images/                    # extracted frames/images (if you generate them)
│   │   ├── 000001.png
│   │   ├── 000002.png
│   │   ├── 000003.png
│   │   └── ...
│   │
│   └── captions.csv               # text captions aligned with images
│
└── metadata/
    ├── version.txt                # caption-1
    ├── source.txt                 # none-yet___anime-captions (dataset ID)
    ├── revision.txt               # 2f1272a94691fd3c8dede0a3697057ab1d4d2296
    ├── schema.json                # {"image": "binary", "text": "string"}
    ├── stats.json                 # optional analytics
    └── preview_samples/
        ├── sample_01.png
        ├── sample_02.png
        └── sample_03.png


```
- **You can use the dataset directly from the HuggingFace website without downloaded it**- 
- **The data contain images and text(the caption belong to the image)**


## Tech stark 

### Backend

- **FastAPI – High-performance async API framework**
- **Uvicorn (expected) – ASGI server**
- **CORS Middleware – Secure cross-origin access**

### Machine Learning & Deep Learning
#### Frameworks
- **PyTorch – Primary deep learning framework**
- **TorchVision – Image preprocessing & pretrained models**
- **TensorFlow (optional) – Disabled OneDNN optimizations for compatibility**
- **Modeling & Vision**
- **HuggingFace Transformers**
- **BLP (BLIPProcessor + BLIPForConditionalGeneration) for image captioning**
- **OpenCV (cv2) – Image decoding & processing**
- **Pillow (PIL) – Image manipulation**
- **AnimeHeadDetector (custom) – Anime character face detection**
- **Faiss – High-speed vector similarity search**

### Recommendation System
#### A hybrid engine combining:
- **Embedding similarity (cosine/Faiss)**
- **Metadata-based filtering & scoring**
- **Fuzzy name matching**
- **Vector-based hybrid scoring**
- **Synopsis embedding matching**

### Custom modules include:
- **load_anime_data, preprocess_anime_data, embedding_sypnosis**
- **hybrid_recommend_by_each, hybrid_recommend_vectorized**
- **metadata_similarity, recommend_for_each_favorite**

### Data & Utilities
- **Pandas – Dataset loading and manipulation**
- **NumPy – Numerical computing**
- **Pathlib / OS / Shutil – File management**
- **TQDM – Progress visualization**
- **Base64 – Image encoding**
- **Argparse – CLI handling**
- **SimpleNamespace – Lightweight configuration wrappers**

### System Integration
- **FAISS index loading/saving**
- **Dynamic dataset image uploads**
- **Preprocessing pipeline for captions, metadata, and embeddings**

## Installation
```
git clone
cd ANIME-APP
```

## Model Details

### Recomendation 

**This system run divede in to two part:**
#### Metadata Filtering
-**The first parts is metadata filtering, search for anime that have the same genres, type, episodes count, studios, duration, rating, score and producers with the original anime**
-**Each element used by the search system is assigned a specific weight. Elements with higher weights have a greater influence on the final results, reflecting their relative importance in the ranking process.**
-**Using jaccard similarity score to compare elements that have string value like genres, scaled similarity for elements that have numeric value type.**
-**After comparing, multiply the elements score with their respective weights And calculate the total for the final score**
-**This score will determine how similar the anime is to the original anime**.

#### Embeddings similarity
-**The second parts is embeddings similarity, using FAISS to embeddings anime sypnosis and compare to other animes.**
-**After embeddings sypnosis into high-dimensional vectors, using Cosine similarity to measures how similar two vectors are based on their direction, the higher the cosine similarity, the more similar the embeddings are.**

#### Hybrid system
-**After calculate the metadata score and the cosin score, computes a hybrid score by combining two normalized score lists after multiply is with an alpha to determine with scores is more important than the other.** 
![alt text](test/test2/recommended%20equation.PNG)

-**The final score will determine which anime is chosen for the recommendation**


## Character classification
-**The system predicts the character in the anime image by cropping the head of the character and feed it into classification model**

### Characters Head detection
-**Using the head detector repository on github to detect the head of the characters**
-**This repository use Yolo-V3 to detect multiple character head in an image**

**Github: https://github.com/grapeot/AnimeHeadDetector**

### Characters Classification

**Choosing one of the five model below after evalutate their performance for the main model**

-**A Vision Transformer (ViT) neural network that applies the transformer architecture directly to image patches, enabling powerful, scalable image understanding without traditional convolutional layers.**

-**ResNet-50 a 50-layer deep convolutional neural network that uses residual connections to make training very deep models easier and more accurate.**

-**EfficientNet-B0 a convolutional neural network that achieves strong accuracy with low computational cost by uniformly scaling depth, width, and resolution using a compound scaling method.**

## Caption and tagging 

### Caption
-**BLIP (Bootstrapping Language–Image Pre-training) a multimodal model that learns to connect images and text using a mix of captioning, image–text matching, and contrastive learning, enabling tasks like caption generation, visual question answering, and image–text retrieval.**

### Tagging
**The tagging system can tags up to 500 tag label using a multi-label models**
**Choosing one of the three model below after evalutate their performance for the main model**
-**ResNet-50 a 50-layer deep convolutional neural network that uses residual connections to make training very deep models easier and more accurate**
-**Danboruu_Resnet50 is a pretrain model using the danbooru2018 and can predicts up to 6000 tags**
-**ResNet-152 a very deep convolutional neural network with 152 layers that uses residual (skip) connections to enable efficient training and achieve strong performance on complex image recognition can classification tasks.**

## Model evaluation

### Recommended system
-**The system recommened anime that in the same franchise with the original anime or have some of the same sypnosis**
![alt text](/test/test2/recommend_gundam.PNG)
-**Recommend for football anime**
![alt text](/test/test2/sports.PNG)

### Characters Head detection
-**The head detection can detect most of the face in the anime images**

![alt text](/test/New%20folder/76121l.jpg)

-**It can can handle the case where the face is barely visible.**

![alt text](/test/test2/164032.jpg)

-**Although all the training data was labeled on a color anime set, the algorithm generalizes well on line drawings.**

![alt text](/test/test2/Capture.PNG)

![alt text](/test/test2/detect_no_color2.PNG)

### Character classification
-**Model Evaluation base on train/test/val dataset**

```
                         Resnet50   EfficientNetB0  Vision_Transformer  

    Train Accuracy        0.9318        0.9258        0.9437        
    Validation Accuracy   0.7758        0.7680        0.8386        
    Test Accuracy         0.7544        0.7896        0.8403      
     
```

-**Because the Vision Transformer model achieves the best performance, we selected it as the main model**

### Caption model
```
Step	Training Loss
50000	0.462100
50500	0.465500
51000	0.455700
51500	0.462300
52000	0.456800
52500	0.454600
53000	0.455200
53500	0.455300
54000	0.447800
54500	0.446600
55000	0.449800
55500	0.446300
56000	0.455700
56500	0.443500
57000	0.442600
57500	0.441300
58000	0.432600
58500	0.441000
59000	0.442600
59500	0.437600
60000	0.435400
60500	0.422800
61000	0.430300
61500	0.430200
62000	0.418800
62500	0.425700
63000	0.419300
63500	0.336500
64000	0.288400
64500	0.282700
65000	0.282600
65500	0.283100
66000	0.286200
66500	0.278200
67000	0.284300
67500	0.279900
68000	0.281800
68500	0.275100
69000	0.278300
69500	0.271400
70000	0.275300
70500	0.272500
71000	0.271900
71500	0.272900
72000	0.260300
72500	0.268800
73000	0.266600
73500	0.266200
74000	0.262700
74500	0.262800
75000	0.259000
75500	0.261300
76000	0.260100
76500	0.259700
77000	0.256800
77500	0.249300
78000	0.253500
78500	0.250900
79000	0.246000
79500	0.250200
80000	0.250100
```
- **The model generate caption that decribe the anime images**

![alt text](/test/test2/caption.PNG)

### Tagging model
-**Model Evaluation base on train/test/val dataset**
```
                         Resnet50   Danboruu_Resnet50  Resnet152 

    Precision             0.4487        0.7445           0.2950        
    Recall                0.5667        0.8549           0.4734        
    F1 Score              0.4774        0.7959           0.3419      
     
```
-**Train loss and validation loss for Resnet152**

![alt text](/test/test2/Resnet152.PNG)
-**We can see that the train loss it reduce but the val lost increase after each epoch, this implies the model is overfitting**

-**Because the Danboruu_Resnet50 pretrain model achieves the best performance, we selected it as the main model for tagging task**

## Instruction how to use the repo
-**Working directory is ANIME-APP**
-**After cloning the repository**
-**Activate backend**
```
cd src
uvicorn api:app --reload
```
-**Open front-end in another terminal**

```
cd anime-dashboard
npm run dev
```

## Front-end
-**Three tab for different task**

![alt text](/test/test2/tab.PNG)

-**List of favourites anime**

![alt text](/test/test2/Favourite.PNG)

**Get recommend button to get recommend**

-**Search anime based on genre and name**

![alt text](/test/test2/search.PNG)

-**List of anime**

![alt text](/test/test2/List.PNG)
**The star symbol to add new favourite anime**

-**Anime character classification**

![alt text](/test/test2/classification_tab.PNG)

-**Anime Taggs and caption**

![alt text](/test/test2/Tagging_and_caption.PNG)

## Referance
-**https://github.com/grapeot/AnimeHeadDetector**
-**https://gwern.net/danbooru2021**
