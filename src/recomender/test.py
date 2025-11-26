import sys
import os
import pandas as pd
from pathlib import Path
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
base = Path(__file__).resolve().parent
sys.path.append(base.parent.parent.as_posix())
from src.recomender.dataset import load_anime_data, preprocess_anime_data, parse_duration, preprocess_anime_data_sypnosis
from src.recomender.embeddings_similarity.embeddings_similarity import hybrid_recommend_by_each, embedding_sypnosis, recommend_by_name_fuzzy, recommend_by_names
from src.recomender.metadata_filtering.metadata_filtering import metadata_similarity_hybrid, metadata_similarity,recommend_for_each_favorite
from src.utils.utils import minmax_normalize, load_embeddings, find_closest_title
import numpy as np
import faiss

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    df_animme_path = base.parent.parent / "data" / "Recomended_Anime_data" / "raw" / "MyAnimeList-Database-master" / "data" / "anime.csv"
    df_anime_sypnosis_path = base.parent.parent / "data" / "Recomended_Anime_data" / "raw" / "MyAnimeList-Database-master" / "data" / "anime_with_synopsis.csv"
    df_anime = pd.read_csv(df_animme_path)
    df_anime = preprocess_anime_data(df_anime)
    df_anime_sypnosis = pd.read_csv(df_anime_sypnosis_path)
    df_anime_sypnosis = preprocess_anime_data_sypnosis(df_anime_sypnosis)

    Name_to_idx = {Name: i for i, Name in enumerate(df_anime_sypnosis["Name"])}
    embedding_path = base.parent / "embeddings_similarity" / "anime_embeddings.npy"
    embeddings = np.load(embedding_path).astype("float32")
    # Load FAISS index
    index_path = base.parent / "embeddings_similarity" / "anime_index.faiss"
    index = faiss.read_index(index_path)

    user_likes = ["Kono Subarashii Sekai ni Shukufuku wo!", "Naruto","Love Live! School Idol Project", "Initial D Fifth Stage","Code Geass: Boukoku no Akito 2 - Hikisakareshi Yokuryuu" ]

    sections = hybrid_recommend_by_each(df_anime= df_anime, df_anime_sypnosis= df_anime_sypnosis, Names = user_likes, k=15, alpha=0.5, mode ="hybrid", embeddings = embeddings, index = index, Name_to_idx = Name_to_idx)


    for liked, recs in sections.items():
        print(f"\nBecause you liked {liked}:\n")
        print(recs)
        
    user_likes = ["Love Live! School Idol Project", "Monster","Initial D Fifth Stage","Code Geass: Boukoku no Akito 2 - Hikisakareshi Yokuryuu"]
    recommendations = recommend_for_each_favorite(df_anime, user_likes, top_n=10)
    for key,recommendation in recommendations.items():
        print(f"Recommendations based on your favorite: {key}")    
        for Name, score, explanation in recommendation:
            print(f"{Name} (similarity: {score:.3f}) → {explanation}")
        print("\n")

    queries = ["Love Live! School Idol Project", "Naruto", "Monster","Code Geass: Boukoku no Akito 2 - Hikisakareshi Yokuryuu"]
    recommendations = recommend_by_names(queries, k=15, df_anime_sypnosis = df_anime_sypnosis, df_anime = df_anime, Name_to_idx=Name_to_idx, embeddings=embeddings, index=index)

    for liked, recs in recommendations.items():
        print(f"\nBecause you liked {liked}:")
        print(recs)
                