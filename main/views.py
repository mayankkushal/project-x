from django.shortcuts import render
from django.conf import settings
from rest_framework.views import APIView
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import load_model
import pickle

from sklearn.externals import joblib
# model = joblib.load('models/sentiment/_model.pkl')

max_review_length = 500
graph = tf.get_default_graph()

# loading tokenizer
with open('models/sentiment/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def _load_model():
    """
        Project specific -> returns the loaded model
    """
    return joblib.load('models/sentiment/_model.pkl')

model = _load_model()

class SentimentAnalysisView(APIView):
    def get(self, request, format=None):
        return Response({"details": "Welcome to sentiment analysis! Project-X"})

    def dispatch(self, request, *args, **kwargs):
        if not settings.DEBUG:
            if request.META.get('HTTP_X_FORWARDED_FOR') not in settings.WHITELIST_IPS.split(','):
                raise PermissionDenied("Not Allowed")
        return super(SentimentAnalysisView, self).dispatch(request, *args, **kwargs)

    def _predict(self):
        """
            Prediction logic goes here.
        """
        self.text = tokenizer.texts_to_sequences([self.text])
        self.text = sequence.pad_sequences(self.text, maxlen=max_review_length)
        global graph
        with graph.as_default():
            predict = model.predict(self.text)
            return predict

    def _get_response(self):
        """
            Converts the prediction into a dict which can directly be passed
            as response.
            `Returns dict()`
        """
        prediction = self._predict()
        return {'score':str(prediction[0][0])}


    def post(self, request, format=None):
        """
            `text`: text that needs to analyzed
        """
        self.text = request.data.get('text')
        if self.text: 
            return Response(self._get_response(), status=status.HTTP_201_CREATED)
        return Response({'details': "text not found"}, status=status.HTTP_400_BAD_REQUEST)