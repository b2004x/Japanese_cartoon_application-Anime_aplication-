import numpy as np
import faiss
from rapidfuzz import process
def minmax_normalize(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)




def load_embeddings(path,df_anime):
    Name_to_idx = {Name: i for i, Name in enumerate(df_anime["Name"])}
    embeddings = np.load(r"G:\hoc\private\Anime\src\recomender\embeddings_similarity\anime_embeddings.npy").astype("float32")

    # Load FAISS index
    index = faiss.read_index(r"G:\hoc\private\Anime\src\recomender\embeddings_similarity\anime_index.faiss")
    return embeddings, index, Name_to_idx



def find_closest_title(query, titles, limit=1, score_cutoff=70):
    matches = process.extract(query, titles, limit=limit, score_cutoff=score_cutoff)
    print(matches[0][0])
    if not matches:
        return None
    return matches[0][0]

def save_faiss_index(index, path):
    embeddings = np.load("path").astype("float32")
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    d = embeddings.shape[1]  # embedding dimension
    index = faiss.IndexFlatIP(d)  # Inner Product works with normalized vectors
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, "anime_index.faiss")

def minmax_normalize(arr):
    arr = np.array(arr, dtype=np.float32)
    if arr.max() == arr.min():  # avoid division by zero
        return np.ones_like(arr)  # or zeros, depending on your preference
    return (arr - arr.min()) / (arr.max() - arr.min())
