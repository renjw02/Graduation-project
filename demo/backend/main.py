from flask import Flask, request
from flask_cors import CORS

from Config import DevelopmentConfig
from predict import model_predict, execute

config = DevelopmentConfig()
app = Flask(__name__)
cors = CORS(app)
app.config.from_object(config)

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
    # print(data)
    question = data['question']
    dataset = data['dataset']

    pred, id, cost_time = model_predict(question, device=app.config['DEVICE'], dataset=dataset)

    if pred is None:
        return {
            'pql': None,
            'results': None,
            'cols': None,
            'cost_time': cost_time,
            'success': False,
        }, 200

    print('pred', pred)

    results, cols = execute(pred, id, dataset, question)
    success = True
    if results is None:
        success = False

    print('cols', cols)
    print('results', results)

    return {
        'pql': pred.replace('table_', 'table_'+id),
        'results': results,
        'cols': cols,
        'cost_time': cost_time,
        'success': success,
    }, 200

if __name__ == '__main__':
    app.run(host=app.config['IP'], debug=True)