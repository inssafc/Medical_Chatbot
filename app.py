from flask import Flask, render_template, request, jsonify

from chat import get_response

from chat import chatbot_response

app = Flask(__name__ ,static_folder='../static/',template_folder='../templates')

@app.get("/")
def index_get():
    return render_template('base.html')


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    response = chatbot_response(text)
    message = {"answer" : response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)