import os
import boto3
from flask import Flask, request
import joblib
from io import BytesIO


class ModelContainer:
    def __init__(self):
        self.loaded = False
        self.model = None

    def get_bucket_and_key(self, model_url):
        split = model_url.split(":")
        cloud_provider = split[0]

        path = split[1][2:].split("/")
        bucket_name = path.pop(0)
        key = "/".join(path)

        return bucket_name, key


    def get_model_from_s3(self, model_url):

        bucket_name, key = self.get_bucket_and_key(model_url)

        session = boto3.Session(
            aws_access_key_id=os.getenv("ACCESS_KEY"),
            aws_secret_access_key=os.getenv("SECRET_KEY"),
        )

        s3 = session.resource('s3')

        with BytesIO() as data:
            s3.Bucket(bucket_name).download_fileobj(key, data)
            data.seek(0)
            model = joblib.load(data)
            self.model = model

    def load(self):
        model_url = os.getenv("MODEL_URL")
        self.get_model_from_s3(model_url)
        self.loaded = True


model_container = ModelContainer()
model_container.load()

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    req = request.json
    data = req["data"]

    if not model_container.loaded:
        model_container.load()

    prediction = model_container.model.predict(data)
    print("prediction:", prediction.tolist())
    return {"prediction": prediction.tolist()}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

