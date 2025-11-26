import sys
import os
import pandas as pd
import numpy as np
module_path = r"G:\hoc\private\Anime"
sys.path.append(module_path)
from recomender.dataset import load_anime_data, preprocess_anime_data, parse_duration
## Hybrid metadata similarity function
from utils.utils import minmax_normalize, load_embeddings, find_closest_title
def jaccard_similarity(set_a, set_b):
    set_a = set(str(set_a).split(", "))
    set_b = set(str(set_b).split(", "))
    if not set_a or not set_b: return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def scaled_similarity(a, b):
    if pd.isna(a) or pd.isna(b): return 0.0
    return 1 / (1 + abs(a - b))

def metadata_similarity_hybrid(idx_a, idx_b, df_anime,weights=None):
    row_a, row_b = df_anime.iloc[idx_a], df_anime.iloc[idx_b]


    if weights is None:
        weights = {
            "Genres": 0.35,
            "Type": 0.1,
            "Episodes": 0.1,
            "Studios": 0.1,
            "Source": 0.1,
            "Duration": 0.1,
            "Rating": 0.05,
            "Score": 0.1,
            "Producers": 0.05,
        }

    sims = {}

    sims["Genres"]   = jaccard_similarity(row_a["Genres"], row_b["Genres"]) ##similarity between two sets of genres
    sims["Type"]     = 1.0 if row_a["Type"] == row_b["Type"] else 0.0  ##exact match of type (TV, Movie, OVA, etc.)
    sims["Episodes"] = scaled_similarity(row_a["Episodes"], row_b["Episodes"]) ## closeness in number of episodes
    sims["Studios"]  = 1.0 if row_a["Studios"] == row_b["Studios"] else 0.0  ## exact match of studio
    sims["Source"]   = 1.0 if row_a["Source"] == row_b["Source"] else 0.0 ## exact match of source material (Manga, Light Novel, Original, etc.)
    sims["Duration"] = scaled_similarity(row_a["Duration"], row_b["Duration"]) ## closeness in episode duration (in minutes)
    sims["Rating"]   = 1.0 if row_a["Rating"] == row_b["Rating"] else 0.0 ## exact match of content rating (PG, R, etc.)
    sims["Score"]    = scaled_similarity(row_a["Score"], row_b["Score"]) ## closeness in community score (0-10 scale)
    sims["Producers"] = 1.0 if row_a["Producers"] == row_b["Producers"] else 0.0 ## match of producers
    total = sum(weights[k] * sims[k] for k in sims)

    return total  # simple average, can be weighted


## sigle metadata similarity function


def metadata_similarity(anime_a, anime_b, weights=None):
    ##the weights control how much each metadata feature contributes to the overall similarity.
    ##Genres (0.35) → Most important, since genre overlap usually drives what anime feels similar.
    ##Type (0.1) → TV vs Movie vs OVA matters, but less than genres.
    ##Episodes (0.1) → People often care about length (short vs long series).
    ##Studios (0.1) → Some studios have distinct styles (e.g. ufotable, Kyoto Animation).
    ##Source (0.1) → Fans of manga adaptations may prefer other manga adaptations.
    ##Duration (0.1) → Per-episode length matters a bit (5 min vs 24 min)
    ##Rating (0.05) → Lower impact, but prevents mismatches (e.g. “kids anime” vs “R+ gore”).
    ##Score (0.1) → Ensures recommended shows are close in community rating.

    if weights is None:
        weights = {
            "Genres": 0.35,
            "Type": 0.1,
            "Episodes": 0.1,
            "Studios": 0.1,
            "Source": 0.1,
            "Duration": 0.1,
            "Rating": 0.05,
            "Score": 0.1,
            "Producers": 0.05,
        }

    sims = {}

    sims["Genres"]   = jaccard_similarity(anime_a["Genres"], anime_b["Genres"]) ##similarity between two sets of genres
    sims["Type"]     = 1.0 if anime_a["Type"] == anime_b["Type"] else 0.0  ##exact match of type (TV, Movie, OVA, etc.)
    sims["Episodes"] = scaled_similarity(anime_a["Episodes"], anime_b["Episodes"]) ## closeness in number of episodes
    sims["Studios"]  = 1.0 if anime_a["Studios"] == anime_b["Studios"] else 0.0  ## exact match of studio
    sims["Source"]   = 1.0 if anime_a["Source"] == anime_b["Source"] else 0.0 ## exact match of source material (Manga, Light Novel, Original, etc.)
    sims["Duration"] = scaled_similarity(anime_a["Duration"], anime_b["Duration"]) ## closeness in episode duration (in minutes)
    sims["Rating"]   = 1.0 if anime_a["Rating"] == anime_b["Rating"] else 0.0 ## exact match of content rating (PG, R, etc.)
    sims["Score"]    = scaled_similarity(anime_a["Score"], anime_b["Score"]) ## closeness in community score (0-10 scale)
    sims["Producers"] = 1.0 if anime_a["Producers"] == anime_b["Producers"] else 0.0 ## match of producers
    total = sum(weights[k] * sims[k] for k in sims)
    return total, sims

def recommend_by_metadata(anime_df, liked_names, top_n=10):
    liked_anime = anime_df[anime_df["Name"].isin([liked_names])]
    if anime_df["Name"].isin([liked_names]).sum() == 0:
        liked_anime = anime_df[anime_df["English name"].isin([liked_names])]
    recs = []

    for idx, candidate in anime_df.iterrows():
        if candidate["Name"] in liked_names:
            continue  # skip already liked

        scores = []
        breakdowns = []

        for _, liked in liked_anime.iterrows():
            sim, sims_detail = metadata_similarity(liked, candidate)
            scores.append(sim)

            # Build explanation
            reasons = []
            if sims_detail["Genres"] > 0:
                reasons.append(f"shares Genres ({sims_detail['Genres']:.2f})")
            if sims_detail["Type"] == 1.0:
                reasons.append("same type")
            if sims_detail["Source"] == 1.0:
                reasons.append("same source material")
            if sims_detail["Rating"] == 1.0:
                reasons.append("same age rating")
            if sims_detail["Studios"] == 1.0:
                reasons.append("same studio")
            if sims_detail["Episodes"] > 0.8:
                reasons.append("similar episode count")
            if sims_detail["Duration"] > 0.8:
                reasons.append("similar episode duration")
            if sims_detail["Score"] > 0.8:
                reasons.append("similar user score")
            if sims_detail["Producers"] == 1.0:
                reasons.append("same producer")

            breakdowns.append(", ".join(reasons))

        if scores:
            avg_score = sum(scores) / len(scores)
            explanation = " | ".join(breakdowns[:2])  # show 2 strongest matches
            recs.append((candidate["Name"], avg_score, explanation))

    recs = sorted(recs, key=lambda x: x[1], reverse=True)

    return recs[:top_n]

def recommend_for_each_favorite(anime_df, favorites, top_n=5):
    results = {}
    for fav in favorites:
        if fav not in anime_df["English name"].values and fav not in anime_df["Name"].values:
            print(f"Dont have {fav} in the dataset")
            continue  # skip if anime not in dataset
        # Get recommendations for this one favorite
        
        
        recs = recommend_by_metadata(anime_df, fav, top_n=top_n)  # <-- your existing function
        results[fav] = recs
    return results


#######################################
import numpy as np

def build_metadata_matrix(df_anime):
    # Episodes normalized
    episodes = df_anime["Episodes"].fillna(0).to_numpy(dtype=np.float32)
    episodes_norm = (episodes - episodes.min()) / (episodes.max() - episodes.min() + 1e-6)

    # Duration normalized
    duration = df_anime["Duration"].str.extract(r'(\d+)').fillna(0).astype(float)[0].to_numpy()
    duration_norm = (duration - duration.min()) / (duration.max() - duration.min() + 1e-6)

    # Type encoded
    type_map = {t: i for i, t in enumerate(df_anime["Type"].dropna().unique())}
    type_encoded = df_anime["Type"].map(type_map).fillna(-1).to_numpy(dtype=np.float32)

    # Genres multi-hot
    all_genres = sorted({g for row in df_anime["Genres"].dropna() for g in row.split(",")})
    genre_map = {g: i for i, g in enumerate(all_genres)}
    genre_matrix = np.zeros((len(df_anime), len(all_genres)), dtype=np.float32)
    for idx, row in enumerate(df_anime["Genres"].fillna("")):
        for g in row.split(","):
            g = g.strip()
            if g in genre_map:
                genre_matrix[idx, genre_map[g]] = 1.0

    # Combine into one metadata matrix
    meta_matrix = np.column_stack([episodes_norm, duration_norm, type_encoded.reshape(-1,1), genre_matrix])
    return meta_matrix


def metadata_similarity_vectorized(meta_matrix, liked_idx, candidate_indices=None):
    """
    Return similarity scores for all candidate_indices (or all if None)
    using cosine similarity.
    """
    liked_vec = meta_matrix[liked_idx:liked_idx+1]
    if candidate_indices is None:
        candidate_matrix = meta_matrix
        candidate_indices = np.arange(meta_matrix.shape[0])
    else:
        candidate_matrix = meta_matrix[candidate_indices]

    # Cosine similarity
    norm_liked = np.linalg.norm(liked_vec, axis=1, keepdims=True)
    norm_candidates = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
    sim = (liked_vec @ candidate_matrix.T).flatten() / (norm_liked.flatten() * norm_candidates.flatten() + 1e-6)

    # Remove self
    mask = candidate_indices != liked_idx
    candidate_indices = candidate_indices[mask]
    sim = sim[mask]

    return candidate_indices, sim
