import sys
import os
import pandas as pd
from pathlib import Path
base = Path(__file__).resolve().parent
module_path = base.parent.parent.parent.as_posix()
if module_path not in sys.path:
    sys.path.append(module_path)
from src.recomender.dataset import load_anime_data, preprocess_anime_data, parse_duration
import numpy as np
from src.recomender.metadata_filtering.metadata_filtering import metadata_similarity_hybrid, metadata_similarity, build_metadata_matrix, metadata_similarity_vectorized
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.utils import minmax_normalize, load_embeddings, find_closest_title,minmax_normalize


# def hybrid_recommend_by_each(Names, k=5, alpha=0.2, df_aniem_sypnosis = None , df_anime = None):

#     embeddings, index, Name_to_idx = load_embeddings(r"G:\hoc\private\Anime\src\recomender\embeddings_similarity\anime_embeddings.npy",df_aniem_sypnosis)
#     """
#     Recommend anime separately for each liked anime title.
#     Returns a dictionary: {liked_title: recommendations_df}
#     """
#     sections = {}
    
#     for liked_title in Names:
#         # Match to dataset index
#         if liked_title not in df_anime["English name"].values and liked_title not in df_anime["Name"].values:
#             print(f"Dont have {liked_title} in the dataset")
#             continue 
#         liked_idx = Name_to_idx[liked_title]
        
#         # Get embedding for this anime
#         anime_vec = embeddings[liked_idx:liked_idx+1]
#         faiss.normalize_L2(anime_vec)

#         # FAISS search
#         scores, indices = index.search(anime_vec, k+50)
#         indices, scores = indices[0], scores[0]
        
#         results = []
#         for idx, s in zip(indices, scores):
#             if idx == liked_idx:  # skip the anime itself
#                 continue
            
#             # Metadata similarity (to just this anime)
#             meta_sim = metadata_similarity_hybrid(idx, liked_idx, df_anime)
            
#             # Hybrid score
#             hybrid_score = alpha * s + (1 - alpha) * meta_sim
#             results.append((idx, hybrid_score))
        
#         # Rank top-k for this anime
#         results = sorted(results, key=lambda x: x[1], reverse=True)[:k]
        
#         rec_df = df_anime.iloc[[r[0] for r in results]][["Name", "Score", "Genres"]].copy()
#         rec_df["hybrid_score"] = [r[1] for r in results]
        
#         sections[liked_title] = rec_df.reset_index(drop=True)
    
#     return sections


# def hybrid_recommend_by_each(
#     Names, 
#     k=5, 
#     alpha=0.5, 
#     mode="hybrid",  # "faiss", "metadata", or "hybrid"
#     df_anime_sypnosis=None, 
#     df_anime=None,
#     embeddings = None,
#     index =None,
#     Name_to_idx = None
# ):
#     """
#     Flexible recommender:
#       - mode="faiss"    → Synopsis-only
#       - mode="metadata" → Metadata-only
#       - mode="hybrid"   → Combine FAISS + metadata
#     """


#     sections = {}

#     for liked_title in Names:
#         # Find index
#         if liked_title not in df_anime["English name"].values and liked_title not in df_anime["Name"].values:
#             print(f"Dont have {liked_title} in the dataset")
#             continue
#         liked_idx = Name_to_idx[liked_title]

#         # -----------------------
#         # MODE 1: Metadata only
#         # -----------------------
#         if mode == "metadata":
#             results = []
#             for idx, _ in df_anime_sypnosis.iterrows():
#                 if idx == liked_idx:
#                     continue
#                 meta_score, sims_detail = metadata_similarity(df_anime.iloc[liked_idx], df_anime.iloc[idx])
#                 results.append((idx, meta_score, sims_detail))

#             results = sorted(results, key=lambda x: x[1], reverse=True)[:k]

#             rec_df = df_anime_sypnosis.iloc[[r[0] for r in results]][["Name", "Score", "Genres"]].copy()
#             rec_df["similarity"] = [r[1] for r in results]
#             rec_df["explanation"] = [
#                 ", ".join(f"{k}:{v:.2f}" for k, v in r[2].items() if v > 0) for r in results
#             ]
#             sections[liked_title] = rec_df.reset_index(drop=True)
#             continue

#         # Get FAISS neighbors (used in both faiss-only and hybrid)
#         anime_vec = embeddings[liked_idx:liked_idx+1]
#         faiss.normalize_L2(anime_vec)
#         scores, indices = index.search(anime_vec, k+50)
#         indices, scores = indices[0], scores[0]

#         # -----------------------
#         # MODE 2: FAISS only
#         # -----------------------
#         if mode == "faiss":
#             results = []
#             for idx, s in zip(indices, scores):
#                 if idx == liked_idx:
#                     continue
#                 results.append((idx, float(s)))

#             results = sorted(results, key=lambda x: x[1], reverse=True)[:k]

#             rec_df = df_anime_sypnosis.iloc[[r[0] for r in results]][["Name", "Score", "Genres"]].copy()
#             rec_df["similarity"] = [r[1] for r in results]
#             sections[liked_title] = rec_df.reset_index(drop=True)
#             continue

#         # -----------------------
#         # MODE 3: Hybrid
#         # -----------------------

#         s_list = []
#         meta_list = []
#         sims_details = []
#         candidate_indices = []

#         for idx, s in zip(indices, scores):
#             if idx == liked_idx:
#                 continue

#             meta_score, sims_detail = metadata_similarity(df_anime.iloc[liked_idx], df_anime.iloc[idx])
            
#             s_list.append(s)
#             meta_list.append(meta_score)
#             sims_details.append(sims_detail)
#             candidate_indices.append(idx)

#         # Normalize both lists to [0,1]
#         s_norm = minmax_normalize(s_list)
#         meta_norm = minmax_normalize(meta_list)

#         # Compute hybrid scores
#         results = []
#         for i, idx in enumerate(candidate_indices):
#             hybrid_score = alpha * s_norm[i] + (1 - alpha) * meta_norm[i]
#             results.append((idx, hybrid_score, sims_details[i]))
            
#         results = sorted(results, key=lambda x: x[1], reverse=True)[:k]

#         rec_df = df_anime_sypnosis.iloc[[r[0] for r in results]][["Name", "Score", "Genres"]].copy()
#         rec_df["hybrid_score"] = [r[1] for r in results]
#         rec_df["explanation"] = [
#             ", ".join(f"{k}:{v:.2f}" for k, v in r[2].items() if v > 0) for r in results
#         ]
#         sections[liked_title] = rec_df.reset_index(drop=True)
#     print("Done recommending")
#     return sections


import numpy as np
import pandas as pd
from src.utils.utils import minmax_normalize
from src.recomender.metadata_filtering.metadata_filtering import metadata_similarity

def hybrid_recommend_by_each(
    Names, 
    k=5, 
    alpha=0.5, 
    mode="hybrid",  # "faiss", "metadata", or "hybrid"
    df_anime_sypnosis=None, 
    df_anime=None,
    embeddings=None,
    index=None,
    Name_to_idx=None
):
    """
    Fast hybrid recommender: returns a dict of DataFrames keyed by favorite anime.
    """
    sections = {}

    # ---------------------------
    # Precompute metadata features as NumPy array for faster similarity
    # ---------------------------
    # Example: store relevant numeric info as array (Episodes, Duration, Type encoded)
    # Here we just vectorize the metadata_similarity call per favorite
    # (assumes metadata_similarity can take NumPy rows if needed)
    print("Starting to recommend...")
    for liked_title in Names:
        if liked_title not in df_anime["English name"].values and liked_title not in df_anime["Name"].values:
            print(f"Dont have {liked_title} in the dataset")
            continue

        liked_idx = Name_to_idx[liked_title]

        # -----------------------
        # MODE 1: Metadata only
        # -----------------------
        if mode == "metadata":
            # Vectorized similarity
            meta_scores = []
            sims_details = []
            liked_row = df_anime.iloc[liked_idx]

            for row in df_anime.itertuples():
                idx = row.Index
                if idx == liked_idx:
                    continue
                score, detail = metadata_similarity(liked_row, df_anime.iloc[idx])
                meta_scores.append(score)
                sims_details.append(detail)

            # Top-k
            top_idx = np.argsort(meta_scores)[::-1][:k]
            rec_df = df_anime_sypnosis.iloc[[i for i in top_idx]][["Name", "Score", "Genres"]].copy()
            rec_df["similarity"] = [meta_scores[i] for i in top_idx]
            rec_df["explanation"] = [
                ", ".join(f"{key}:{value:.2f}" for key, value in sims_details[i].items() if value > 0)
                for i in top_idx
            ]
            sections[liked_title] = rec_df.reset_index(drop=True)
            continue

        # -----------------------
        # FAISS neighbors (for faiss and hybrid)
        # -----------------------
        anime_vec = embeddings[liked_idx:liked_idx+1]
        faiss.normalize_L2(anime_vec)
        scores, indices = index.search(anime_vec, k+50)
        indices, scores = indices[0], scores[0]

        # -----------------------
        # MODE 2: FAISS only
        # -----------------------
        if mode == "faiss":
            results = [(idx, float(s)) for idx, s in zip(indices, scores) if idx != liked_idx]
            results = sorted(results, key=lambda x: x[1], reverse=True)[:k]

            rec_df = df_anime_sypnosis.iloc[[r[0] for r in results]][["Name", "Score", "Genres"]].copy()
            rec_df["similarity"] = [r[1] for r in results]
            sections[liked_title] = rec_df.reset_index(drop=True)
            continue

        # -----------------------
        # MODE 3: Hybrid
        # -----------------------
        print("Computing hybrid for:", liked_title)
        candidate_indices = []
        s_list = []
        meta_list = []
        sims_details = []

        liked_row = df_anime.iloc[liked_idx]

        for idx, s in zip(indices, scores):
            if idx == liked_idx:
                continue
            candidate_indices.append(idx)
            s_list.append(s)
            meta_score, sim_detail = metadata_similarity(liked_row, df_anime.iloc[idx])
            meta_list.append(meta_score)
            sims_details.append(sim_detail)

        # Normalize
        s_norm = minmax_normalize(s_list)
        meta_norm = minmax_normalize(meta_list)

        # Compute hybrid scores
        hybrid_scores = [alpha*s + (1-alpha)*m for s,m in zip(s_norm, meta_norm)]
        results = sorted(
            [(idx, score, detail) for idx, score, detail in zip(candidate_indices, hybrid_scores, sims_details)],
            key=lambda x: x[1],
            reverse=True
        )[:k]

        rec_df = df_anime_sypnosis.iloc[[r[0] for r in results]][["Name", "Score", "Genres"]].copy()
        rec_df["hybrid_score"] = [r[1] for r in results]
        rec_df["explanation"] = [
            ", ".join(f"{key}:{value:.2f}" for key, value in r[2].items() if value > 0) for r in results
        ]
        sections[liked_title] = rec_df.reset_index(drop=True)

    print("Done recommending")
    return sections




def recommend_by_name_fuzzy(querys, k=5, df_anime_sypnosis = None, df_anime = None, embeddings = None, index =None, Name_to_idx = None):
    results = {}

    for query in querys:
        # Step 1: Find closest match in dataset
        best_match = find_closest_title(query, df_anime["Name"].tolist())
        if best_match is None:
            print(f"No close match found for '{query}'")
            continue

        # Step 2: Lookup index
        anime_idx = Name_to_idx[best_match]

        # Step 3: Build query vector
        qvec = embeddings[anime_idx].reshape(1, -1).astype("float32")
        faiss.normalize_L2(qvec)

        # Step 4: Search FAISS
        scores, indices = index.search(qvec, k + 1)
        indices = indices[0][1:]  # remove the query itself
        scores = scores[0][1:]

        # Step 5: Build result DataFrame
        rec_df = df_anime_sypnosis.iloc[indices][["Name", "Score", "Genres"]].copy()
        rec_df["similarity"] = scores
        results[best_match] = rec_df.reset_index(drop=True)

    return results

def embedding_sypnosis(path):
    df_anime_sypnosis = load_anime_data(path)
    model = SentenceTransformer("paraphrase-mpnet-base-v2")
    embeddings = model.encode(df_anime_sypnosis["sypnopsis"].tolist(), show_progress_bar=True, convert_to_tensor=False)
    np.save(r"G:\hoc\private\Anime\notebooks\Recomended_Anime_data/anime_embeddings.npy", embeddings)



def recommend_by_names(titles, k=5,df_anime=None, df_anime_sypnosis=None, embeddings = None, index =None, Name_to_idx = None):
    results = {}

    for title in titles:
        if title not in Name_to_idx:
            print(f"'{title}' not found in dataset.")
            continue

        # Get index of this anime
        anime_idx = Name_to_idx[title]

        # Build query vector
        query_vec = embeddings[anime_idx].reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_vec)

        # Search FAISS
        scores, indices = index.search(query_vec, k + 1)  # +1 to skip itself
        indices = indices[0][1:]  # drop the query itself
        scores = scores[0][1:]

        # Build result DataFrame
        rec_df = df_anime_sypnosis.iloc[indices][["Name", "Score", "Genres"]].copy()
        rec_df["similarity"] = scores
        results[title] = rec_df.reset_index(drop=True)

    return results




def hybrid_recommend_vectorized(
    Names,
    k=5,
    alpha=0.5,
    mode="hybrid",
    df_anime=None,
    df_anime_sypnosis=None,
    embeddings=None,
    index=None,
    Name_to_idx=None
):
    sections = {}
    meta_matrix = build_metadata_matrix(df_anime)

    for liked_title in Names:
        if liked_title not in df_anime["English name"].values and liked_title not in df_anime["Name"].values:
            print(f"Dont have {liked_title} in the dataset")
            continue

        liked_idx = Name_to_idx[liked_title]

        # ----------------- FAISS neighbors -----------------
        anime_vec = embeddings[liked_idx:liked_idx+1]
        faiss.normalize_L2(anime_vec)
        scores, indices = index.search(anime_vec, k+50)
        indices, scores = indices[0], scores[0]

        candidate_indices = indices[indices != liked_idx]
        scores = scores[indices != liked_idx]

        # ----------------- MODE 1: Metadata only -----------------
        if mode == "metadata":
            candidate_indices, meta_scores = metadata_similarity_vectorized(meta_matrix, liked_idx)
            top_idx = np.argsort(meta_scores)[::-1][:k]
            rec_indices = candidate_indices[top_idx]
            rec_scores = meta_scores[top_idx]

            rec_df = df_anime_sypnosis.iloc[rec_indices][["Name","Score","Genres"]].copy()
            rec_df["similarity"] = rec_scores
            rec_df["explanation"] = [""]*len(rec_df)  # can keep detailed explanation if needed
            sections[liked_title] = rec_df.reset_index(drop=True)
            continue

        # ----------------- MODE 2: FAISS only -----------------
        if mode == "faiss":
            top_idx = np.argsort(scores)[::-1][:k]
            rec_indices = candidate_indices[top_idx]
            rec_scores = scores[top_idx]

            rec_df = df_anime_sypnosis.iloc[rec_indices][["Name","Score","Genres"]].copy()
            rec_df["similarity"] = rec_scores
            sections[liked_title] = rec_df.reset_index(drop=True)
            continue

        # ----------------- MODE 3: Hybrid -----------------
        # Vectorized metadata for candidates
        _, meta_scores = metadata_similarity_vectorized(meta_matrix, liked_idx, candidate_indices)

        # Normalize
        s_norm = minmax_normalize(scores)
        meta_norm = minmax_normalize(meta_scores)

        hybrid_scores = alpha*s_norm + (1-alpha)*meta_norm
        top_idx = np.argsort(hybrid_scores)[::-1][:k]
        rec_indices = candidate_indices[top_idx]
        rec_hybrid_scores = hybrid_scores[top_idx]

        rec_df = df_anime_sypnosis.iloc[rec_indices][["Name","Score","Genres"]].copy()
        rec_df["hybrid_score"] = rec_hybrid_scores
        rec_df["explanation"] = [""]*len(rec_df)  # can add detailed explanation if needed
        sections[liked_title] = rec_df.reset_index(drop=True)

    return sections
