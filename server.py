
from flask import Flask,render_template,url_for,request
import tensorflow as tf
import tensorflow_text as text
from official.nlp import optimization 


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost/replica:0/task:0/device:CPU:0')
    reloaded_model = tf.keras.models.load_model('FakeNewsData_bert', options=load_options, compile=False)

    txt = []
    txt.append(str(request.form.get('txt', "")))
    my_prediction = tf.sigmoid(reloaded_model(tf.constant(txt)))
    print(float(my_prediction))
    return render_template('results.html',prediction = round(float(my_prediction)))



if __name__ == '__main__':
    print("Starting server...")
    app.run(port=4000, host='0.0.0.0', debug=True)
