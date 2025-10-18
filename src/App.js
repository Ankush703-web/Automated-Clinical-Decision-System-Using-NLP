import React, { useState } from 'react';
import './App.css';
import ReactMarkdown from 'react-markdown';

function App() {
  const [pdf, setPdf] = useState(null);
  const [xray, setXray] = useState(null);
  const [eyeImage, setEyeImage] = useState(null);
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (setter) => (e) => {
    setter(e.target.files[0]);
  };

  const handleQuerySubmit = async () => {
  if (!query.trim()) {
    setResponse('Please enter a medical query.');
    return;
  }

  setIsLoading(true);
  try {
    const formData = new FormData();
    formData.append('query', query);
    
    // Append files if they exist, otherwise append 'no_image_data'
    formData.append('pdf', pdf || 'no_image_data');
    formData.append('xray', xray || 'no_image_data');
    formData.append('eyeImage', eyeImage || 'no_image_data');

    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    setResponse(data.response);
  } catch (error) {
    setResponse('Error: Could not get medical suggestions. Please try again.');
    console.error('Error:', error);
  }
  setIsLoading(false);
};
  return (
    <div className="container py-5 bg-light">
      {/* Header with Stethoscope SVG */}
      <div className="text-center mb-4">
        <img
          src="/images/clintnobg.png"
          alt="CLINT"
          width="701"
          height="701"
          className="mb-3 logo-img"
          style={{ objectFit: 'contain' }}
        />
        <h1 className="text-success mt-2">Clinical Language Intelligence Using NLP & Transformers</h1>
        <p className="text-muted">
          Upload clinical data and ask questions to receive AI-powered suggestions.
        </p>
      </div>


      {/* File Uploads */}
      <div className="card shadow-sm mb-4">
        <div className="card-body">
          <div className="mb-3">
            <label className="form-label text-success">Upload EHR Report (PDF)</label>
            <input type="file" accept="application/pdf" onChange={handleFileChange(setPdf)} className="form-control" />
          </div>

          <div className="mb-3">
            <label className="form-label text-success">Upload X-Ray Image</label>
            <input type="file" accept="image/*" onChange={handleFileChange(setXray)} className="form-control" />
          </div>

          <div className="mb-3">
            <label className="form-label text-success">Upload Eye Image</label>
            <input type="file" accept="image/*" onChange={handleFileChange(setEyeImage)} className="form-control" />
          </div>
        </div>
      </div>

      {/* Clinical Query Input */}
      <div className="card shadow-sm mb-4">
        <div className="card-body">
          <label className="form-label text-success">Enter Clinical Query</label>
          <textarea
            className="form-control mb-3"
            rows="4"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., What diagnosis fits this report?"
          ></textarea>
          <button
            onClick={handleQuerySubmit}
            className="btn btn-success w-100"
            disabled={isLoading}
          >
            {isLoading ? 'Processing...' : '🔍 Analyze & Respond'}
          </button>
        </div>
      </div>

      {/* Response Section */}
      <div className="card shadow-sm">
        <div className="card-body">
          <h5 className="text-success mb-3">Medical Suggestions</h5>
          <div className="border rounded p-3 bg-white">
            {response ? (
              <div style={{ fontFamily: '-apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Roboto' }}>
                <ReactMarkdown>
                  {response}
                </ReactMarkdown>
              </div>
            ) : (
              <p className="text-muted">
                Your medical suggestions will appear here after you submit a query.
                Please note: This is an AI assistant and not a replacement for professional medical advice.
              </p>
            )}
          </div>
          {response && (
            <div className="mt-3">
              <small className="text-muted">
                ⚕️ These are general suggestions. Always consult a healthcare provider for medical advice.
              </small>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
