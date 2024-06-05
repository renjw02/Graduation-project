from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World'


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive a string of data(natural language question) from the client, return a string of pql(query) and a table of results.
    Arguments:
        data: str
    Returns:
        pql: str
        results: list
    """
    data = request.get_json()
    question = data['question']
    return {
        'pql': 'SELECT * FROM table',
        'results': [
            {'name': 'Alice', 'age': 25},
            {'name': 'Bob', 'age': 30}
        ]
    }, 200

if __name__ == '__main__':
    app.run()