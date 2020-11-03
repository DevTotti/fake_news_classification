from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return 'index.html'


def predict('/classify', methods=['GET','POST']):
    if request.method == 'GET':
        return 'index.html'


    elif request.method == 'POST':
        news_content = request.form['news']

        predictedOutcome = mainScript(news_content)

        return ''



if __name__ == '__main__':
    app.run()
