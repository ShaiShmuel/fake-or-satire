
from flask import Flask,render_template,url_for,request
# import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os



app = Flask(__name__)


class NewsDetector:

    def __init__(self, fake_data_path, satire_data_path):
        # define classification
        self.fake_classification = 0
        self.satire_classification = 1

        # load SBERT
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')


        # load lodistic regression model
        self.linear_reg = LinearRegression()

        # load news data
        self.__df_fake = self.__read_txt_files(fake_data_path, self.fake_classification)
        self.__df_satire = self.__read_txt_files(satire_data_path, self.satire_classification)
        self.__df_data = pd.concat([self.__df_fake, self.__df_satire], ignore_index=True)

        print(self.__df_data)

    def split(self, test_size):
        X = np.array(self.__df_data.loc[:, self.__df_data.columns != 'class'])
        y = np.array(self.__df_data['class'])

        # Split to train and test 
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(X, y, test_size = test_size, random_state = 42)

    def fit(self):
        # self.__X_train = self.__X_train.values
        # self.__X_train = np.vstack(self.__X_train[:][:])
        # print(type(self.__X_train))
        self.model = self.linear_reg.fit(self.__X_train, self.__y_train)

    def predict(self, news_text):
        embd_text = self.sbert.encode(news_text)
        cols = []
        for i in range(len(embd_text)):
            column_name = 'dim' + str(i+1)
            cols.append(column_name)

        # Return DataFrame with the encoded text and class
        df_pred = pd.DataFrame(data = embd_text , columns=cols)


        result = self.linear_reg.predict(df_pred)
        return result    



    def __read_txt_files(self,source_folder, binary_classification: int):
        # Initialize an empty list to store the text
        texts = []
        columns = []

        # Iterate through each file
        for file in os.listdir(source_folder):
            # Only consider .txt files
            if file.endswith(".txt"):
                # Open the file
                with open(os.path.join(source_folder, file), "r" ,encoding='ascii',errors='ignore') as f:
                    text = f.read()
                # Check if the file is empty
                if text:
                    # Append the encoded text to the list
                    texts.append(self.sbert.encode(text))
        
        vector_length = len(texts[0])

        for i in range(vector_length):
            column_name = 'dim' + str(i+1)
            columns.append(column_name)

        # Return DataFrame with the encoded text and class
        df = pd.DataFrame(data = texts , columns=columns)
        df['class'] = binary_classification

        return df

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    # load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    # reloaded_model = tf.keras.models.load_model('FakeNewsData_bert', options=load_options)
    # txt = []
    # txt.append(str(request.form.get('txt', "")))
    # my_prediction = tf.sigmoid(reloaded_model(tf.constant(txt)))
    # model = NewsDetector('FakeNewsData/StoryText 2/Fake/finalFake', 'FakeNewsData/StoryText 2/Satire/finalSatire')
    # model.split(0.3)
    # model.fit()
    # print(model.predict('this is a fake news text'))

    return render_template('result.html',prediction = round(float(my_prediction)))



if __name__ == '__main__':
    print("Starting server...")
    # app.run(port=4000, host='0.0.0.0', debug=True)

    model = NewsDetector('FakeNewsData/StoryText 2/Fake/finalFake', 'FakeNewsData/StoryText 2/Satire/finalSatire')
    model.split(0.3)
    model.fit()
    print(model.predict('this is a fake news text'))
    