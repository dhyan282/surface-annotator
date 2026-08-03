frontend/src/App.tsx
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { format } from 'date-fns';
import './App.css';

interface UploadedImage {
  id: string;
  name: string;
  file: File;
  preview: string;
  status: 'pending' | 'uploading' | 'annotating' | 'completed' | 'error';
  progress: number;
  result?: any;
}

interface AnnotationResult {
  preview_path: string;
  road_polys: number;
  walkway_polys: number;
  bike_polys: number;
  car_polys: number;
}

function App() {
  const [images, setImages] = useState<UploadedImage[]>([]);
  const [isAnnotating, setIsAnnotating] = useState(false);
  const [websocketConnected, setWebsocketConnected] = useState(false);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const API_BASE_URL = 'https://your-app-name.netlify.app';

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.hostname}:8080/api/ws`;
    
    const websocket = new WebSocket(wsUrl);
    setWs(websocket);

    websocket.onopen = () => {
      console.log('WebSocket connected');
      setWebsocketConnected(true);
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'annotation_result') {
        setImages(prev => prev.map(img => {
          if (img.id === data.image_id) {
            return {
              ...img,
              status: 'completed',
              progress: 100,
              result: data.result
            };
          }
          return img;
        }));
      }
    };

    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      setWebsocketConnected(false);
    };

    return () => {
      websocket.close();
    };
  }, []);

  const uploadImage = async (file: File): Promise<string> => {
    const formData = new FormData();
    formData.append('image', file);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/annotate`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Upload failed');
      }
      
      const result = await response.json();
      return result.task_id;
    } catch (error) {
      console.error('Error uploading image:', error);
      throw error;
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files) return;

    Array.from(files).forEach(file => {
      if (!file.type.startsWith('image/')) {
        alert('Please upload only image files');
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        const image = new Image();
        image.onload = () => {
          setImages(prev => [...prev, {
            id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
            name: file.name,
            file: file,
            preview: e.target?.result as string,
            status: 'pending',
            progress: 0
          }]);
        };
        image.src = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    });

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const startAnnotation = async () => {
    if (images.length === 0) return;
    
    setIsAnnotating(true);
    setImages(prev => prev.map(img => ({ ...img, status: 'uploading', progress: 0 })));

    for (const image of images) {
      if (image.status !== 'pending') continue;
      
      try {
        setImages(prev => prev.map(img => 
          img.id === image.id ? { ...img, status: 'uploading', progress: 50 } : img
        ));
        
        await uploadImage(image.file);
        
        setImages(prev => prev.map(img => 
          img.id === image.id ? { ...img, status: 'annotating', progress: 75 } : img
        ));
        
      } catch (error) {
        setImages(prev => prev.map(img => 
          img.id === image.id ? { ...img, status: 'error', progress: 0 } : img
        ));
      }
    }
    
    setIsAnnotating(false);
  };

  const removeImage = (id: string) => {
    setImages(prev => prev.filter(img => img.id !== id));
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>
            <span className="gradient-text">Surface Auto-Annotator</span>
          </h1>
          <p>AI-powered paved surface annotation for road, walkway, bikepath & cars</p>
          <div className="status-indicators">
            {websocketConnected && (
              <span className="status-connected">✓ WebSocket Connected</span>
            )}
            <span className="status-api">✓ API Proxy Ready</span>
          </div>
        </div>
      </header>

      <main className="main-content">
        <aside className="sidebar glass-effect">
          <div className="upload-section">
            <h3>Upload Images</h3>
            <p>Drag and drop or click to select files</p>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*"
              onChange={handleFileUpload}
              disabled={isAnnotating}
              className="file-input"
            />
          </div>

          <div className="controls">
            <button 
              onClick={startAnnotation}
              disabled={images.length === 0 || isAnnotating}
              className={`annotate-btn ${isAnnotating ? 'disabled' : ''}`}
            >
              {isAnnotating ? 'Annotating...' : 'Start Annotation'}
            </button>
            
            {images.length > 0 && (
              <div className="image-count">
                {images.filter(img => img.status === 'pending').length} images ready
              </div>
            )}
          </div>

          <div className="progress-section">
            {images.map(img => img.status !== 'pending' && (
              <div key={img.id} className="progress-item">
                <div className="progress-header">
                  <span>{img.name}</span>
                  <span>{img.status}</span>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ width: `${img.progress}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </aside>

        <section className="image-gallery">
          {images.map(image => (
            <div key={image.id} className="image-card">
              <div className="image-preview">
                <img src={image.preview} alt={image.name} />
                <button 
                  onClick={() => removeImage(image.id)}
                  className="remove-btn"
                  title="Remove image"
                >
                  ×
                </button>
              </div>
              <div className="image-info">
                <p className="image-name">{image.name}</p>
                <div className="status-badge status-{image.status}">
                  {image.status}
                </div>
              </div>
              {image.result && (
                <div className="annotation-results">
                  <h4>Annotation Results</h4>
                  <div className="results-grid">
                    <div className="result-item">
                      <span className="result-value">{image.result.road_polys}</span>
                      <span className="result-label">Road</span>
                    </div>
                    <div className="result-item">
                      <span className="result-value">{image.result.walkway_polys}</span>
                      <span className="result-label">Walkway</span>
                    </div>
                    <div className="result-item">
                      <span className="result-value">{image.result.bike_polys}</span>
                      <span className="result-label">Bike Path</span>
                    </div>
                    <div className="result-item">
                      <span className="result-value">{image.result.car_polys}</span>
                      <span className="result-label">Cars</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </section>
      </main>
    </div>
  );
}

export default App;
