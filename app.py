from flask import Flask, render_template, request, jsonify
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

app = Flask(__name__)

# Load GPT-Neo
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json['user_input']
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('response generated')
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
