# 🏥 CLINT – Clinical Language Intelligence Using NLP & Transformers  
### _AI-powered Clinical Query Assistant with Multi-Modal Medical Analysis_  

CLINT is a full-stack medical AI application that allows users to upload **EHR reports (PDFs)**, **X-Ray images**, **Eye images**, and enter any **clinical query**, after which the system processes all inputs and generates structured, evidence-based medical guidance.

---

## 🚀 Features

### 🔍 1. Clinical Query Understanding (NER)
- Extracts medical entities from query text using **Biomedical BERT NER**  
- Detects diseases, drugs, anatomy, symptoms & more

### 📄 2. EHR (PDF) Summarization
- Extracts text from PDF using `PyPDF2`  
- Summarizes using **BART Large CNN**

### 🩻 3. X-Ray Disease Classification
- Upload chest X-Ray  
- TensorFlow (`xray_model_final.h5`) + labels.json  
- Predicts top disease with probability

### 👁️ 4. Eye Disease Detection
- Upload retinal eye image  
- PyTorch ResNet-18 model  
- Loads trained weights from `eye_disease_model.pth`

### 🤖 5. AI-Generated Structured Medical Advice
- Uses `Intelligent-Internet/II-Medical-8B-1706`  
- Output includes:
  - Key Suggestions
  - Lifestyle Modifications
  - When to Seek Medical Attention
  - Warnings & Precautions  
- Delivered in Markdown format

> ⚠️ Disclaimer: This is for educational & research use only.

---

## 🛠 Tech Stack

### Frontend
- React.js  
- Bootstrap  
- react-markdown

### Backend
- Python (Flask)  
- PyTorch + TorchVision  
- TensorFlow/Keras  
- Transformers (HuggingFace)  
- HuggingFace Inference API  
- PIL, PyPDF2  

---

## 📁 Project Structure

📦 CLINT ├── frontend/ │   ├── App.js │   ├── App.css │   └── public/images/clintnobg.png │ ├── backend/ │   ├── server.py │   ├── eye_disease_model.pth │   ├── xray_model_final.h5 │   ├── classes.json │   ├── labels.json │   ├── BERT/models/biomedical-ner-all/ │   ├── BART/models/bart-large-cnn/ │   └── uploads/

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

git clone https://github.com/Ankush703-web/Automated-Clinical-Decision-System-using-NLP.git

cd Automated-Clinical-Decision-System-using-NLP


---

🖥 Backend Setup (Flask)

Install dependencies

pip install -r requirements.txt

Set HuggingFace API Key

export HF_API_KEY="your_api_key_here"

Run Backend

python server.py

Backend runs at:
➡️ http://localhost:5000


---

🌐 Frontend Setup (React)

Install packages

npm install

Start frontend

npm start

Frontend runs at:
➡️ http://localhost:3000


---

🧪 Usage Guide

1. Upload one or more of:

PDF (EHR report)

Chest X-Ray

Eye image


2. Enter your clinical query:

What diagnosis fits this report?

3. Click Analyze & Respond

The system performs:

Medical NER

PDF summarization

X-ray classification

Eye disease detection

LLM reasoning


And returns structured medical advice.


---

📦 API Endpoint

POST /predict

Form-Data Fields:

query: text
pdf: file (optional)
xray: file (optional)
eyeImage: file (optional)

Response JSON:

{
  "status": "success",
  "response": "markdown-formatted medical suggestions"
}


---

📜 License

MIT


---

🤝 Contributing

Pull requests and improvements are welcome.


---

⭐ Show Support

If you found this useful, consider giving this repo a ⭐ on 
