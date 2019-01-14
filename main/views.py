from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status

from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import load_model
import pickle

# model = load_model('my_model.h5')
# print('Model Loaded...')

with open('models/sentiment/model.pickle', 'rb') as m:
    model = pickle.load(m)

max_review_length = 500

# loading tokenizer
with open('models/sentiment/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    print("tokenizer loaded...")
graph = tf.get_default_graph()


class SentimentAnalysisView(APIView):
    def get(self, request, format=None):
        return Response({"details": "Welcome to sentiment analysis! Project-X"})

    def post(self, request, format=None):
        """
            `text`: text that needs to analyzed
        """
        text = request.data.get('text')
        if text:
            text = tokenizer.texts_to_sequences([text])
            text = sequence.pad_sequences(text, maxlen=max_review_length)
            global graph
            with graph.as_default():
                predict = model.predict(text)      
                return Response({'score':str(predict[0][0])}, status=status.HTTP_201_CREATED)
        return Response({'details': "text not found"}, status=status.HTTP_400_BAD_REQUEST)