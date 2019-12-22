import pandas as pd
from flask import Flask, jsonify, request
import pickle

# load model
model = pickle.load(open('model.pkl','rb'))

# model = pickle.loads(saved_model) 

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = model.predict(data_df)
    print("the prediction is",result)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5006, debug=True)