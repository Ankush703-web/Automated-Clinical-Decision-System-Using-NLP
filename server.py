from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the Hugging Face client
client = InferenceClient(
    provider="featherless-ai",
    api_key=os.environ["HF_API_KEY"],
)

# Load from local path of BERT model for biomedical NER
local_model_path ="BERT/models/biomedical-ner-all/" 
med_NER={}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs('./uploads', exist_ok=True)
        
        # Ensure model files exist before processing
        model_path = "eye_disease_model.pth"
        classes_path = "classes.json"
        
        if not os.path.exists(model_path) or not os.path.exists(classes_path):
            return jsonify({'error': 'Required model files are missing'}), 500

        query = request.form.get('query')

        if not query:
            return jsonify({'error': 'No query provided'}), 400

        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForTokenClassification.from_pretrained(local_model_path)

        # Create NER pipeline
        nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

        # Example text
        text = query

        # Run NER
        entities = nlp(text)

        # Display results
        for ent in entities:
            word = ent['word']
            entity_group = ent['entity_group']
            score = round(ent['score'], 3)
            
            med_NER[word] = {
                'entity_group': entity_group,
                'score': score
            }
            print(f"{word} → {entity_group} (score: {score:.3f})")


        # Get files or default value
        pdf = request.files.get('pdf')
        xray = request.files.get('xray')
        eye_image = request.files.get('eyeImage')

        # Process files if they exist
        if pdf and pdf != 'no_image_data':
            # Process PDF file
            pdf.save(f"./uploads/{pdf.filename}")
            
        if xray and xray != 'no_image_data':
            # Process X-ray image
            try:
                x_ray_upload_path = f"./uploads/{xray.filename}"
                xray.save(x_ray_upload_path)
                if not os.path.exists(x_ray_upload_path):
                    raise FileNotFoundError(f"Failed to save image: {x_ray_upload_path}")
            except Exception as e:
                return jsonify({'error': f'Error processing eye image: {str(e)}'}), 500
            
            # Paths
            model_path = "./xray_model_final.h5"
            labels_path = "./labels.json"
            image_path = f"./uploads/{xray.filename}"
            
            # Load model
            model = load_model(model_path)

            # Load labels
            with open(labels_path, "r") as f:
                all_labels = json.load(f)

            IMG_SIZE = 224

            def predict_xray(img_path):
                img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                probs = model.predict(img_array, verbose=0)[0]  # get probabilities
                best_index = np.argmax(probs)                   # find index of highest probability
                best_label = all_labels[best_index]             # get corresponding label
                best_prob = float(probs[best_index])            # convert to float

                return best_label, best_prob

            # ---- Predict on a single image ----
            predicted_label, predicted_prob = predict_xray(image_path)

            print(f"\n🩻 Image: {os.path.basename(image_path)}")
            print(f"Predicted Finding: {predicted_label} ({predicted_prob:.4f})")

        if eye_image and eye_image != 'no_image_data':
            # Process eye image
            try:
                upload_path = f"./uploads/{eye_image.filename}"
                eye_image.save(upload_path)
                if not os.path.exists(upload_path):
                    raise FileNotFoundError(f"Failed to save image: {upload_path}")
            except Exception as e:
                return jsonify({'error': f'Error processing eye image: {str(e)}'}), 500

            # Set device (GPU if available)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load classes
            with open("classes.json", "r") as f:
                classes = json.load(f)
            
            # Rebuild the model architecture (ResNet18 with custom final layer)
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, len(classes))

            # Load the saved weights
            model.load_state_dict(torch.load("eye_disease_model.pth", map_location=device))
            model = model.to(device)
            model.eval()

            #print("✅ Model and classes loaded successfully!")
            #print("Classes:", classes)

            # Define the image transform (same as in your notebook)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            # Prediction function
            def predict_image(img_path):
                # Load and preprocess the image
                image = Image.open(img_path).convert("RGB")
                image = transform(image).unsqueeze(0).to(device)

                # Run inference            
                with torch.no_grad():
                    output = model(image)
                    _, predicted = torch.max(output, 1)

                # Return the predicted class
                return classes[predicted.item()]

            # Predict on a specific eye image
            eye_img = f"./uploads/{eye_image.filename}" 
            eye_disease_prediction = predict_image(eye_img)
            print("Prediction for", eye_img, ":", eye_disease_prediction)

        structured_prompt = f"""
        Medical query analysis using entities(NER):{med_NER} \n
        x_ray_disease: {predicted_label if xray and xray != 'no_image_data' else 'N/A'} \n
        eye_disease: {eye_disease_prediction if eye_image and eye_image != 'no_image_data' else 'N/A'} \n
        
        Based on this query: {query}
        
        if there are no Medical related query found using NER then respond that "No relevant medical information found."  Otherwise,

        Provide medical advice in this exact format:

        ### Key Suggestions for Self-Care
        - Stay hydrated with warm fluids
        - Use honey for cough (adults and children >1 year)
        
        ### Lifestyle Modifications
        - Use a humidifier in bedroom
        - Avoid irritants and allergens
        
        ### When to Seek Medical Attention
        - If symptoms worsen or persist >7 days
        - If experiencing severe symptoms
        
        ### Warnings & Precautions
        - Note about OTC medication safety
        - When to consult healthcare provider

        Keep responses evidence-based and practical.
        Use proper Markdown line breaks between sections.
        """

        completion = client.chat.completions.create(
            model="Intelligent-Internet/II-Medical-8B-1706",
            messages=[{"role": "user", "content": structured_prompt}],
        )

        response_text = completion.choices[0].message.content
        # Preserve line breaks in markdown
        cleaned_response = '\n'.join(
            line for line in response_text.splitlines()
            if not line.strip().startswith('<') and not line.strip().endswith('>')
        )

        response = {
            'status': 'success',
            'response': cleaned_response
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)