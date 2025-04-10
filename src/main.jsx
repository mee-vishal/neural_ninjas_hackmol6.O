import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import { createBrowserRouter, RouterProvider } from "react-router-dom"; // ✅ Updated
import "./index.css";
import Layout from "./components/Layout.jsx";
import VideoUploader from "./components/videoUploader.jsx";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      {
        index: true,
        element: <App />
      }
    ]
  },
  {
    path: "/upload",
    element: <Layout />,
    children: [
      {
        index: true,
        element: <VideoUploader />
      }
    ]
  }
]);

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <RouterProvider router={router} /> 
  </React.StrictMode>
);
