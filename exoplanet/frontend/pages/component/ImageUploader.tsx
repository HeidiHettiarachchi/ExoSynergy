import { useState, useRef } from "react";
import "./ImageUploader.css";

interface ImageUploaderProps {
  onImageUpload: (file: File) => void;
  loading: boolean;
}

export default function ImageUploader({
  onImageUpload,
  loading,
}: ImageUploaderProps) {
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file: File) => {
    if (file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
      onImageUpload(file);
    }
  };

  const onButtonClick = () => {
    inputRef.current?.click();
  };

  return (
    <div className="image-uploader-container">
      <div
        className={`upload-area ${dragActive ? "drag-active" : ""} ${
          loading ? "disabled" : ""
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={!preview ? onButtonClick : undefined}
      >
        <input
          ref={inputRef}
          type="file"
          className="upload-input"
          accept="image/*"
          onChange={handleChange}
          disabled={loading}
        />

        {preview ? (
          <div className="preview-container">
            <div className="preview-image-wrapper">
              <img src={preview} alt="Preview" className="preview-image" />
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onButtonClick();
              }}
              disabled={loading}
              className="upload-button"
            >
              {loading ? "Processing..." : "Upload Different Image"}
            </button>
          </div>
        ) : (
          <div className="empty-state">
            <div className="upload-icon-wrapper">
              <div className="upload-icon">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
              </div>
            </div>
            <div className="upload-text">
              <h3>Drop your mineral image here</h3>
              <p>or click to browse from your device</p>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onButtonClick();
              }}
              disabled={loading}
              className="upload-button"
              style={{ maxWidth: "300px", margin: "0 auto" }}
            >
              {loading ? "Processing..." : "Select Image"}
            </button>
            <p className="upload-hint">
              Supported formats: JPG, PNG, WEBP (Max 10MB)
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
