import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import Papa from "papaparse";

export default function Details() {
  const { id } = useParams();
  const [anime, setAnime] = useState(null);

  useEffect(() => {
    Papa.parse("/anime.csv", {
      download: true,
      header: true,
      complete: (results) => {
        const found = results.data.find((a) => a.MAL_ID === id);
        setAnime(found);
      },
    });
  }, [id]);

  if (!anime) {
    return <p className="text-center text-gray-500">Loading details...</p>;
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <Link to="/" className="text-blue-500 underline">
        ← Back
      </Link>

      <div className="bg-white shadow rounded-xl p-6 mt-4">
        <div className="flex flex-col md:flex-row gap-6">
          <img
            src={anime.main_pic}
            alt={anime.Name}
            className="w-64 rounded-lg shadow"
          />
          <div>
            <h1 className="text-3xl font-bold mb-2">{anime.Name}</h1>
            <ul className="space-y-1 text-gray-700">
              <li><b>Score:</b> {anime.Score}</li>
              <li><b>Genres:</b> {anime.Genres}</li>
              <li><b>English name:</b> {anime["English name"]}</li>
              <li><b>Japanese name:</b> {anime["Japanese name"]}</li>
              <li><b>Type:</b> {anime.Type}</li>
              <li><b>Episodes:</b> {anime.Episodes}</li>
              <li><b>Aired:</b> {anime.Aired}</li>
              <li><b>Premiered:</b> {anime.Premiered}</li>
              <li><b>Producers:</b> {anime.Producers}</li>
              <li><b>Studios:</b> {anime.Studios}</li>
              <li><b>Source:</b> {anime.Source}</li>
              <li><b>Duration:</b> {anime.Duration}</li>
              <li><b>Rating:</b> {anime.Rating}</li>
              <li><b>sypnopsis:</b> {anime.sypnopsis}</li>
              <li><b>MAL:</b> {anime.anime_url}</li>
              <li><b>Ranked:</b> {anime.Ranked}</li>
              <li><b>Popularity:</b> {anime.Popularity}</li>
              <li><b>Members:</b> {anime.Members}</li>
              <li><b>Watching:</b> {anime.Watching}</li>
              <li><b>Completed:</b> {anime.Completed}</li>
              <li><b>Favorites:</b> {anime.Favorites}</li>
              <li><b>Dropped:</b> {anime.Dropped}</li>

            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
