import sys
import os
import pandas as pd
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
module_path = r"G:\hoc\private\Anime"
if module_path not in sys.path:
    sys.path.append(module_path)
from src.recomender.dataset import load_anime_data, preprocess_anime_data, parse_duration, preprocess_anime_data_sypnosis
from src.recomender.embeddings_similarity.embeddings_similarity import hybrid_recommend_by_each, embedding_sypnosis, recommend_by_name_fuzzy, recommend_by_names, hybrid_recommend_vectorized
from src.recomender.metadata_filtering.metadata_filtering import metadata_similarity_hybrid, metadata_similarity,recommend_for_each_favorite
from src.utils.utils import minmax_normalize, load_embeddings, find_closest_title
import numpy as np
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS

def normalize_name(name):
    return name.lower().strip()
    
# Initialize Flask
app = Flask(__name__)
CORS(app)  # allow requests from React

# ----------------------------
# Preload data once at startup
# ----------------------------
print("Loading datasets...")
df_anime = pd.read_csv(
    r"G:\hoc\private\Anime\data\Recomended_Anime_data\raw\MyAnimeList-Database-master\data\anime.csv"
)
df_anime = preprocess_anime_data(df_anime)

df_anime_sypnosis = pd.read_csv(
    r"G:\hoc\private\Anime\data\Recomended_Anime_data\raw\MyAnimeList-Database-master\data\anime_with_synopsis.csv"
)
df_anime_sypnosis = preprocess_anime_data_sypnosis(df_anime_sypnosis)

Name_to_idx = {Name: i for i, Name in enumerate(df_anime_sypnosis["Name"])}

embeddings = np.load(
    r"G:\hoc\private\Anime\src\recomender\embeddings_similarity\anime_embeddings.npy"
).astype("float32")

index = faiss.read_index(
    r"G:\hoc\private\Anime\src\recomender\embeddings_similarity\anime_index.faiss"
)

print("Data loaded. Ready to accept requests.")

# ----------------------------
# Optional cache to speed up repeated requests
# ----------------------------
recommendation_cache = {}

# ----------------------------
# Recommendation endpoint
# ----------------------------
@app.route("/recommend", methods=["GET"])
def get_recommendations():
    print("Received recommendation request")
    names_param = request.args.get("names")
    if not names_param:
        return jsonify({"error": "names parameter required"}), 400
    print(names_param)

    # Normalize user input names
    user_likes = [name for name in names_param.split(",")]
    cache_key = ",".join(user_likes)

    # Return cached result if available
    if cache_key in recommendation_cache:
        return jsonify({"recommendations": recommendation_cache[cache_key]})
    print(f"Generating recommendations for: {user_likes}")
    # Call the recommender (top-k reduced for speed)
    sections = hybrid_recommend_by_each(
        df_anime=df_anime,
        df_anime_sypnosis=df_anime_sypnosis,
        Names=user_likes,
        k=10,  # smaller k = faster
        alpha=0.5,
        mode="hybrid",
        embeddings=embeddings,
        index=index,
        Name_to_idx=Name_to_idx
    )   
    for liked, recs in sections.items():
        print(f"\nBecause you liked {liked}:\n")
        print(recs)

    # Convert DataFrames to JSON-friendly format
    response = {}
    for liked, df_recs in sections.items():
        response[liked] = df_recs[
            ["Name", "Score", "Genres", "hybrid_score", "explanation"]
        ].to_dict(orient="records")

    # Cache the result
    recommendation_cache[cache_key] = response

    return jsonify({"recommendations": response})

# ----------------------------
# Run the Flask app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
    
    # user_likes = ["Kono Subarashii Sekai ni Shukufuku wo!", "Naruto","Love Live! School Idol Project", "Initial D Fifth Stage","Code Geass: Boukoku no Akito 2 - Hikisakareshi Yokuryuu" ]

    # sections = hybrid_recommend_by_each(df_anime= df_anime, df_anime_sypnosis= df_anime_sypnosis, Names = user_likes, k=15, alpha=0.5, mode ="hybrid", embeddings = embeddings, index = index, Name_to_idx = Name_to_idx)


    # for liked, recs in sections.items():
    #     print(f"\nBecause you liked {liked}:\n")
    #     print(recs)
        
    # user_likes = ["Love Live! School Idol Project", "Monster","Initial D Fifth Stage","Code Geass: Boukoku no Akito 2 - Hikisakareshi Yokuryuu"]
    # recommendations = recommend_for_each_favorite(df_anime, user_likes, top_n=10)
    # for key,recommendation in recommendations.items():
    #     print(f"Recommendations based on your favorite: {key}")    
    #     for Name, score, explanation in recommendation:
    #         print(f"{Name} (similarity: {score:.3f}) → {explanation}")
    #     print("\n")

    # queries = ["Love Live! School Idol Project", "Naruto", "Monster","Code Geass: Boukoku no Akito 2 - Hikisakareshi Yokuryuu"]
    # recommendations = recommend_by_names(queries, k=15, df_anime_sypnosis = df_anime_sypnosis, df_anime = df_anime, Name_to_idx=Name_to_idx, embeddings=embeddings, index=index)

    # for liked, recs in recommendations.items():
    #     print(f"\nBecause you liked {liked}:")
    #     print(recs)
                