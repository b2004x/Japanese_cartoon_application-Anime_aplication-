import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Papa from "papaparse";

export default function Home() {
  const [animeList, setAnimeList] = useState([]);
  const [search, setSearch] = useState("");
  const [selectedGenres, setSelectedGenres] = useState([]);
  const [genres, setGenres] = useState([]);
  const [favorites, setFavorites] = useState(() => {
    const saved = localStorage.getItem("favorites");
    return saved ? JSON.parse(saved) : [];
  });
  const [recommendations, setRecommendations] = useState({});
  const [loading, setLoading] = useState(false); // ⏳ loading state

  // 📥 Load anime.csv
  useEffect(() => {
    Papa.parse("/anime.csv", {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        const cleaned = results.data.filter(
          (row) => row.MAL_ID && row.Name && row.main_pic
        );

        const allGenres = new Set();
        cleaned.forEach((row) => {
          if (row.Genres) {
            row.Genres.split(",").forEach((g) => allGenres.add(g.trim()));
          }
        });

        setGenres([...allGenres].sort());
        setAnimeList(cleaned);
      },
    });
  }, []);

  // 💾 Save favorites
  useEffect(() => {
    localStorage.setItem("favorites", JSON.stringify(favorites));
  }, [favorites]);

  // 🎯 Fetch recommendations
  const getRecommendations = () => {
    if (favorites.length === 0 || animeList.length === 0) {
      console.warn("Anime list or favorites not ready yet.");
      return;
    }

    setLoading(true); // start loading
    setRecommendations({}); //

    const favAnime = animeList.filter((a) => favorites.includes(a.MAL_ID));
    const favNames = favAnime.map((a) => a.Name);

    fetch(
      `http://127.0.0.1:8000/recommend?names=${encodeURIComponent(
        favNames.join(",")
      )}`
    )
      .then((res) => res.json())
      .then((data) => {
        console.log("Backend response:", data);
        if (data.recommendations) {
          setRecommendations(data.recommendations);
        }
      })
      .catch((err) => console.error(err))
      .finally(() => setLoading(false)); // stop loading
  };

  const toggleFavorite = (id) => {
    setFavorites((prev) =>
      prev.includes(id) ? prev.filter((fid) => fid !== id) : [...prev, id]
    );
  };

  const toggleGenre = (genre) => {
    setSelectedGenres((prev) =>
      prev.includes(genre) ? prev.filter((g) => g !== genre) : [...prev, genre]
    );
  };

  // 🔍 Filter anime
  const filteredAnime = animeList.filter((anime) => {
    const matchesSearch = anime.Name.toLowerCase().includes(
      search.toLowerCase()
    );
    const matchesGenres =
      selectedGenres.length === 0 ||
      (anime.Genres && selectedGenres.every((g) => anime.Genres.includes(g)));
    return matchesSearch && matchesGenres;
  });

  const favoriteAnime = animeList.filter((anime) =>
    favorites.includes(anime.MAL_ID)
  );

  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-5xl font-extrabold text-center mb-10 text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-purple-600 drop-shadow-lg">
        Anime Dashboard
      </h1>

      {/* 💖 Favorites Section */}
      {favoriteAnime.length > 0 && (
        <div className="mb-16">
          <h2 className="text-3xl font-bold mb-6 border-b-4 border-yellow-400 inline-block">
            ⭐ Your Favorites
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {favoriteAnime.map((anime) => (
              <div
                key={anime.MAL_ID}
                className="relative bg-white rounded-2xl shadow-lg hover:shadow-2xl transform hover:-translate-y-2 transition-all duration-300 overflow-hidden"
              >
                <button
                  onClick={() => toggleFavorite(anime.MAL_ID)}
                  className="absolute top-3 right-3 bg-red-500 hover:bg-red-600 text-white rounded-full px-3 py-1 text-sm shadow-md"
                >
                  ✕
                </button>

                <Link to={`/anime/${anime.MAL_ID}`}>
                  <img
                    src={anime.main_pic}
                    alt={anime.Name}
                    className="rounded-t-2xl w-full h-56 object-cover"
                  />
                  <h2 className="mt-3 text-center font-bold text-lg px-2">
                    {anime.Name}
                  </h2>
                </Link>
              </div>
            ))}
          </div>

          <div className="mt-8 text-center">
            <button
              onClick={getRecommendations}
              disabled={loading}
              className={`px-8 py-3 rounded-full shadow-lg text-lg font-semibold transition ${
                loading
                  ? "bg-gray-400 cursor-not-allowed"
                  : "bg-gradient-to-r from-blue-500 to-purple-600 hover:from-purple-600 hover:to-blue-500 text-white"
              }`}
            >
              {loading ? "⏳ Loading..." : "🎯 Get Recommendations"}
            </button>
          </div>
        </div>
      )}

      {/* 🧠 Recommendations */}
      {recommendations && Object.keys(recommendations).length > 0 && (
        <div className="mb-16">
          <h2 className="text-3xl font-bold mb-6 border-b-4 border-blue-400 inline-block">
            🎯 Recommended For You
          </h2>

          {Object.entries(recommendations).map(([favName, recs]) => (
            <div
              key={favName}
              className="mb-12 p-6 rounded-2xl bg-gradient-to-r from-gray-50 to-gray-100 shadow-lg"
            >
              <h3 className="text-2xl font-semibold mb-6 text-gray-800">
                Because you liked{" "}
                <span className="text-purple-600">{favName}</span>:
              </h3>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                {recs.map((anime, idx) => {
                  const match = animeList.find((a) => a.Name === anime.Name);
                  const isFav = match && favorites.includes(match.MAL_ID);

                  return (
                    <div
                      key={idx}
                      className="relative bg-white rounded-2xl shadow-md hover:shadow-2xl transform hover:-translate-y-2 transition-all duration-300 overflow-hidden"
                    >
                      {match && (
                        <button
                          onClick={() => toggleFavorite(match.MAL_ID)}
                          className={`absolute top-3 right-3 text-2xl ${
                            isFav
                              ? "text-yellow-400 drop-shadow-md"
                              : "text-gray-400 hover:text-yellow-400"
                          }`}
                        >
                          ★
                        </button>
                      )}

                      <Link to={match ? `/anime/${match.MAL_ID}` : "#"}>
                        <img
                          src={match ? match.main_pic : "/fallback.jpg"}
                          alt={anime.Name}
                          className="rounded-t-2xl w-full h-56 object-cover"
                        />
                        <h2 className="mt-3 text-center font-bold text-lg px-2">
                          {anime.Name}
                        </h2>
                        <p className="text-sm text-gray-500 text-center">
                          {anime.Genres}
                        </p>
                        <p className="text-sm text-gray-400 text-center">
                          Score: {anime.Score ?? "N/A"}
                        </p>
                      </Link>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* 🔍 Search Bar */}
      <div className="flex justify-center mb-8">
        <input
          type="text"
          placeholder="🔍 Search by name..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="px-5 py-3 rounded-full border border-gray-300 focus:ring-4 focus:ring-blue-400 w-full md:w-2/3 lg:w-1/2 shadow-sm"
        />
      </div>

      {/* 🎭 Genre Filter */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mb-10">
        {genres.map((g) => (
        <label key={g} className="genre-tag">
          <input
            type="checkbox"
            checked={selectedGenres.includes(g)}
            onChange={() => toggleGenre(g)}
          />
          {g}
        </label>
        ))}
      </div>

      {/* 🎴 Anime Grid */}
      {filteredAnime.length === 0 ? (
        <p className="text-center text-gray-500">No anime found.</p>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {filteredAnime.map((anime) => {
            const isFav = favorites.includes(anime.MAL_ID);
            return (
              <div
                key={anime.MAL_ID}
                className="relative bg-white rounded-2xl shadow-md hover:shadow-2xl transform hover:-translate-y-2 transition-all duration-300 overflow-hidden"
              >
                <button
                  onClick={() => toggleFavorite(anime.MAL_ID)}
                  className={`absolute top-3 right-3 text-2xl ${
                    isFav
                      ? "text-yellow-400 drop-shadow-md"
                      : "text-gray-400 hover:text-yellow-400"
                  }`}
                >
                  ★
                </button>

                <Link to={`/anime/${anime.MAL_ID}`}>
                  <img
                    src={anime.main_pic}
                    alt={anime.Name}
                    className="rounded-t-2xl w-full h-56 object-cover"
                  />
                  <h2 className="mt-3 text-center font-bold text-lg px-2">
                    {anime.Name}
                  </h2>
                </Link>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
