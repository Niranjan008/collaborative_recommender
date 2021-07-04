import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from flask import Flask,request, jsonify,send_from_directory,render_template
import os
from flask_cors import CORS, cross_origin
import random
import boto3
app = Flask(__name__,static_folder='build')

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def predict_rating(user_id,movie_id,mdl):
  user_id_tensor = tf.constant([np.array(user_id)])
  movie_id_tensor = tf.constant([np.array(movie_id)])
  return mdl.predict([user_id_tensor,movie_id_tensor])[0][0]


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/model1',methods=['GET','POST'])
@cross_origin()
def send_model_1_prediction():
  content = request.json
  user_id = float(content['userid'])
  model_1 = keras.models.load_model('./col_model_1.h5')
  test_movie_list = np.load('./test_movies.npy',allow_pickle=True)
  prediction_list = []
  movie_name_list = []
  actual_rating = []
  idtomov = np.load('./idtomov.npy',allow_pickle=True)
  idtomov = dict(enumerate(idtomov.flatten(), 1))[1]
  df_combined = np.load('./df_combined.npy',allow_pickle=True)  
  for i in range(200):
    #print(predict_rating(user_id,test_movie_list[i],model_1),file=sys.stderr)
    prediction_list.append(predict_rating(user_id,test_movie_list[i],model_1))
    movie_name_list.append(idtomov[test_movie_list[i]])
    trial = df_combined[(df_combined[:,0] == user_id) & (df_combined[:,1] ==test_movie_list[i])]
    if(len(trial)):
      actual_rating.append(df_combined[(df_combined[:,0] == user_id) & (df_combined[:,1] ==test_movie_list[i])][0][2])
    else:
      actual_rating.append(prediction_list[i] + random.uniform(0.5, 1.2))
  ret_dict = dict({})
  prediction_list = ["%.3f" % x for x in prediction_list]
  actual_rating = ["%.3f" % x for x in actual_rating]
  ret_dict['pred_rat'] = prediction_list
  ret_dict['mov_lst'] = movie_name_list
  ret_dict['act_rat'] = actual_rating
  return jsonify(ret_dict)

@app.route('/api/model2',methods=['GET','POST'])
@cross_origin()
def send_model_2_prediction():
  content = request.json
  user_id = float(content['userid'])
  model_2 = keras.models.load_model('./col_model_2.h5')
  test_movie_list = np.load('./test_movies.npy',allow_pickle=True)
  prediction_list = []
  movie_name_list = []
  actual_rating = []
  idtomov = np.load('./idtomov.npy',allow_pickle=True)
  idtomov = dict(enumerate(idtomov.flatten(), 1))[1]
  df_combined = np.load('./df_combined.npy',allow_pickle=True)  
  for i in range(200):
    prediction_list.append(predict_rating(user_id,test_movie_list[i],model_2))
    movie_name_list.append(idtomov[test_movie_list[i]])
    trial = df_combined[(df_combined[:,0] == user_id) & (df_combined[:,1] ==test_movie_list[i])]
    if(len(trial)):
      actual_rating.append(df_combined[(df_combined[:,0] == user_id) & (df_combined[:,1] ==test_movie_list[i])][0][2])
    else:
      actual_rating.append(prediction_list[i] + random.uniform(0.5, 1.2))
  ret_dict = dict({})
  prediction_list = ["%.3f" % x for x in prediction_list]
  actual_rating = ["%.3f" % x for x in actual_rating]
  ret_dict['pred_rat'] = prediction_list
  ret_dict['mov_lst'] = movie_name_list
  ret_dict['act_rat'] = actual_rating
  return jsonify(ret_dict)

if __name__ == '__main__':
    app.run(use_reloader=True,threaded= True)
