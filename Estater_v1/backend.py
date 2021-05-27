from flask import Flask, request, render_template, jsonify
import valution_func
import os
import json

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = dict(request.form)
        n_data, i_data = valution_func.get_intermed_data(data)
        return render_template('intermed.html', data=n_data,i_data=i_data)
    else:
        with open("encodings.json", "r") as f:
            data = json.load(f)
        return render_template('index.html', purp=data["purpose"], types=data["type"], agree=data["agreement"], cls=data["class"], seller=data["seller_status"], op=data["op"])


@app.route('/value', methods=['POST'])
def value():
    data = request.get_json()
    data = valution_func.value_property(data)
    return jsonify(data)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))