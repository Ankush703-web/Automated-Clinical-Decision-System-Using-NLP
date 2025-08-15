from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the Hugging Face client
client = InferenceClient(
    provider="featherless-ai",
    api_key=os.environ["HF_API_KEY"],
)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        query = data.get('query')

        if not query:
            return jsonify({'error': 'No query provided'}), 400

        structured_prompt = f"""
        Based on this query: {query}

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