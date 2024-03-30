from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the trained model and tokenizer
model_path = './trained_model'  # Adjust if your model is saved elsewhere
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.eos_token_id
model.eval()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    question = data.get('text', '')
    context = data.get('context', None)  # Extract context from the request
    temperature = float(data.get('temperature', 1.0))
    max_length = int(data.get('max_length', 50))
    generated_text = generate_response(question, context=context, temperature=temperature, max_length=max_length)
    return jsonify({'generated_text': generated_text})

def generate_response(question, context=None, temperature=1.0, max_length=100):
    formatted_prompt = f"### Question: {question}\n### Answer:"
    if context:
        formatted_prompt = f"{context}\n{formatted_prompt}"
    input_ids = tokenizer.encode(formatted_prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, temperature=temperature, top_p=0.9, do_sample=True)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer_start = generated_text.find("### Answer:") + len("### Answer:")
    generated_answer = generated_text[answer_start:].strip()
    return generated_answer

if __name__ == '__main__':
    app.run(debug=True)
