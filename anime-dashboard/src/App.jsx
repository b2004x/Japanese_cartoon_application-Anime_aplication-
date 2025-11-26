import React from "react";
import { Routes, Route , Link} from "react-router-dom";
import Home from "./pages/Home";
import Details from "./pages/Details";
import Classifier from "./pages/Classifier";
import Tagger from "./pages/Tagger"; 
import "./App.css";

export default function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="flex justify-center gap-10 py-6 bg-gray-900 shadow-lg mb-10">
        <Link
          to="/"
          className="nav-tab"
        >
          Home
        </Link>

        <Link
          to="/classify"
          className="nav-tab blue"
        >
          Classify Anime Characters
        </Link>

        <Link
          to="/tagger"
          className="nav-tab green"
        >
          Tag & Caption
        </Link>
      </nav>



      {/* App Routes */}
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/anime/:id" element={<Details />} />
        <Route path="/classify" element={<Classifier />} />
        <Route path="/tagger" element={<Tagger />} />
      </Routes>
    </div>
  );
}