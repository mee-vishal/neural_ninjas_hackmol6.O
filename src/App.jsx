import React from "react";
import { MapContainer, TileLayer, Circle, useMapEvents } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";
import { Link, useNavigate, NavLink, redirect } from "react-router-dom";
import Header from "./components/Header";

function LocationMarker() {
  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;
      console.log(`Latitude: ${lat}\nLongitude: ${lng}`);
    },
  });
  return null;
}

function App() {
  const position = [31.394543488902787, 75.53321578006086]; // Delhi
  const navigate = useNavigate()
  return (
    <div className="container">
      <div className="map">
        <MapContainer
          center={position}
          zoom={30}
          style={{ height: "100vh", width: "100%" }}
        >
          <TileLayer
            attribution="&copy; OpenStreetMap contributors"
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <Circle
            center={position}
            radius={5}
            pathOptions={{ color: "red", fillColor: "#f03", fillOpacity: 0.5 }}
          />
          <LocationMarker />
        </MapContainer>
      </div>

      <div className="popUp">
        <h2>Want to contribute?</h2>
        <button onClick={() => navigate("/upload")}>Click Here</button>
      </div>
    </div>
  );
}

export default App;
