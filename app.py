from flask import Flask, request, render_template
import boto3
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
import pickle
from dotenv import load_dotenv, find_dotenv
from werkzeug.utils import secure_filename
import os
from io import StringIO

load_dotenv(find_dotenv())
app = Flask(__name__)

s3 = boto3.client('s3',
                  aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
                  aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
bucket_name = os.getenv('AWS_BUCKET_NAME')


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', upload_message="", train_message="", predictions=[])


@app.route('/upload', methods=['post'])
def upload():
    if request.method == 'POST':
        img = request.files['file']
        if img:
            file_name = secure_filename(img.filename)
            img.save(file_name)
            s3.upload_file(Bucket=bucket_name, Filename=file_name, Key=file_name)
    return render_template("index.html", upload_message="Upload completed!", train_message="", predictions=[])


@app.route("/train_model", methods=['POST'])
def train():
    # Training dataset
    object_key = 'train.csv'
    csv_obj = s3.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    data = pd.read_csv(StringIO(csv_string))

    # Separate features and target variable(s)
    train_x = data.drop(columns=['Item_Outlet_Sales'])
    train_y = data['Item_Outlet_Sales']

    model_pipeline = pipeline()

    model_pipeline.fit(train_x, train_y)
    model_pipeline.predict(train_x)

    save_model(model_pipeline)
    return render_template("index.html", upload_message="", train_message="Training Completed!", predictions=[])


@app.route("/test_model", methods=['POST'])
def test():
    result = test_model()

    return render_template("index.html", upload_message="", train_message="", predictions=result)


def test_model():
    object_key = 'test.csv'
    csv_obj = s3.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    test_data = pd.read_csv(StringIO(csv_string))

    response = s3.get_object(Bucket=bucket_name, Key='model.sav')

    body = response['Body'].read()
    pickled_model = pickle.loads(body)
    result = pickled_model.predict(test_data)

    return result


def pipeline():
    pre_process = ColumnTransformer(remainder='passthrough',
                                    transformers=[('drop_columns', 'drop', ['Item_Identifier',
                                                                            'Outlet_Identifier',
                                                                            'Item_Fat_Content',
                                                                            'Item_Type',
                                                                            'Outlet_Identifier',
                                                                            'Outlet_Size',
                                                                            'Outlet_Location_Type',
                                                                            'Outlet_Type'
                                                                            ]),
                                                  ('impute_item_weight', SimpleImputer(strategy='mean'),
                                                   ['Item_Weight']),
                                                  ('scale_data', StandardScaler(), ['Item_MRP'])])

    # The pipeline: Get the outlet binary columns -> Pre process the dataset -> Train the Model
    model_pipeline = Pipeline(steps=[('get_outlet_binary_columns', OutletTypeEncoder()),
                                     ('pre_processing', pre_process),
                                     ('random_forest', RandomForestRegressor(max_depth=10, random_state=2))])

    return model_pipeline


class OutletTypeEncoder(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, documents, y=None):
        return self

    def transform(self, x_dataset):
        x_dataset['outlet_grocery_store'] = (x_dataset['Outlet_Type'] == 'Grocery Store') * 1
        x_dataset['outlet_supermarket_3'] = (x_dataset['Outlet_Type'] == 'Supermarket Type3') * 1
        x_dataset['outlet_identifier_OUT027'] = (x_dataset['Outlet_Identifier'] == 'OUT027') * 1

        return x_dataset


def save_model(model_pipeline):
    file_name = 'model.sav'
    pickle.dump(model_pipeline, open(file_name, 'wb'))
    s3.upload_file(Bucket=bucket_name, Filename=file_name, Key=file_name)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
