# app.py
from flask import Flask, request, jsonify, render_template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os

app = Flask(__name__)

# Load API key from environment variable
api_key = os.environ.get("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.9, google_api_key=api_key)
app.static_folder = 'static'
chat_history = []

def generate_prompt(chat_history, input_text):
    messages = [
        ("system", "You are a Critical thinker where by you are have knowlegde on barbet coin everything you need to know about barbet coin is this History of Barbet coins... [Your provided context here] ...")
    ]
    messages.extend(chat_history)
    messages.append(("human", input_text))
    
    return ChatPromptTemplate.from_messages(messages)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    input_text = request.json['message']
    prompt = generate_prompt(chat_history, input_text)
    chain = prompt | model
    response = chain.invoke({"input": input_text})
    chat_history.append(("human", input_text))
    chat_history.append(("ai", response.content))
    return jsonify({'response': response.content})

if __name__ == '__main__':
    app.run(debug=True,port=5005)