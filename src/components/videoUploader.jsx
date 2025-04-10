import React, { useState } from "react";
import axios from "axios";

const VideoUploader = () => {
  const [videoUrl, setVideoUrl] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");

  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploading(true);
    setMessage("");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("upload_preset", "ml_upload"); // your Cloudinary unsigned preset

    try {
      // Step 1: Upload to Cloudinary
      const res = await axios.post(
        "https://api.cloudinary.com/v1_1/dx3chjisf/video/upload",
        formData
      );

      const cloudinaryUrl = res.data.secure_url;
      setVideoUrl(cloudinaryUrl);
      console.log("Uploaded video URL:", cloudinaryUrl);

      // Step 2: Send to backend to store in Firestore
      await axios.post("https://pothole-server.onrender.com/store-video", {
        url: cloudinaryUrl,
      });

      setMessage("✅ Video uploaded and saved to database!");
    } catch (error) {
      console.error("Upload failed", error);
      setMessage("❌ Upload failed. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="videoContainer p-4 max-w-md mx-auto bg-white shadow-md rounded-xl">
      <h1 className="head1">
        Let's Fill <span className="head2">Potholes Together</span>
      </h1>
      <div className="form">
      <div className="input">
        <label htmlFor="from">From</label>
        <input type="text" name="from" id="from" />
      </div>
      <div className="input">
        <label htmlFor="to">To</label>
        <input type="text" name="to" id="to" />
      </div>
      <h2 className="text-xl font-semibold mb-2">Upload a Video</h2>
      <input type="file" accept="video/*" onChange={handleUpload} />
      {uploading && <p className="text-yellow-600 mt-2">Uploading...</p>}
      {message && <p className="mt-2">{message}</p>}
      {videoUrl && (
        <div className="mt-4">
          <video src={videoUrl} controls width="100%" className="rounded-md" />
          <p className="text-sm text-gray-500 break-words mt-2">{videoUrl}</p>
        </div>
      )}
      </div>
    </div>
  );
};

export default VideoUploader;
