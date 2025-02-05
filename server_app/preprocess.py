import numpy as np
import pandas as pd
import librosa
import pickle
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import load_model

def feature_extractor_40(audio_file):
    # 확장자가 .wav 또는 .mp3인지 확인
    if audio_file.endswith('.wav') or audio_file.endswith('.mp3'):
        audio, sample_rate = librosa.load(audio_file)
        mffcs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_features_40 = np.mean(mffcs_features.T, axis=0)
        return mfccs_features_40
    else:
        return None  # 올바르지 않은 파일 확장자 처리

def feature_extractor_80(audio_file):
    # 확장자가 .wav 또는 .mp3인지 확인
    if audio_file.endswith('.wav') or audio_file.endswith('.mp3'):
        audio, sample_rate = librosa.load(audio_file)
        mffcs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_features_80 = np.mean(mffcs_features.T, axis=0)
        return mfccs_features_80
    else:
        return None  # 올바르지 않은 파일 확장자 처리

def scaler1(mfcc_features):
    scaler_path = './artifacts/first_model_scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    mfcc_features_scaled = scaler.transform([mfcc_features])
    
    return mfcc_features_scaled

def scaler2(mfcc_features):
    scaler_path = './artifacts/second_model_scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    mfcc_features_scaled = scaler.transform([mfcc_features])
    
    return mfcc_features_scaled

def differential_mfcc(mfcc_value):
    mfcc_features = np.array(mfcc_value)  # Pandas 값 변환 수정
    delta_mfccs = librosa.feature.delta(mfcc_features, order=1)         # ΔMFCC (1차 미분, 40개) 계산
    delta2_mfccs = librosa.feature.delta(mfcc_features, order=2)        # ΔΔMFCC (2차 미분, 40개) 계산
    
    concat_features = np.hstack((mfcc_features, delta_mfccs, delta2_mfccs))  # (샘플 수, 120)
    
    return concat_features
    
def process1(audio):
    mfcc1 = feature_extractor_40(audio)
    if mfcc1 is None:
        return None
    mfcc2 = scaler1(mfcc1)
    mfcc3 = differential_mfcc(mfcc2)
    
    return mfcc3

def process2(audio):
    mfcc1 = feature_extractor_80(audio) 
    if mfcc1 is None:
        return None
    mfcc2 = differential_mfcc(mfcc1)
    mfcc3 = scaler1(mfcc2)
    
    return mfcc3


def model1(mfcc_features):
    model_path = './artifacts/first_model_sim1.keras'
    model = load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU})  # 모델 로드
    
    pred = model.predict(np.array([mfcc_features]))  # 예측 수행
    pred_class = (pred > 0.5).astype(int)  # 확률값을 이진 분류로 변환
    
    if pred_class == 0:
        print('병원X')
    else:
        print('병원O')
        
    return pred_class

def model2(mfcc_features):
    model_path = './artifacts/second_model_sim2_mfcc80.keras'
    model = load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU}) 
    label_path = './artifacts/second_model_label_encoder.pkl'
    with open(label_path, 'rb') as f:
        label_encoder = pickle.load(f)

    pred = model.predict(np.array([mfcc_features]))
    pred_class_index = np.argmax(pred)
    pred_class_name = label_encoder.inverse_transform([pred_class_index])[0]

    if pred_class_index == 0:
        print('awake, 깸')
    elif pred_class_index == 1:
        print('diaper, 귀저기갈아야함')
    elif pred_class_index == 2:
        print('hug, 안아줘야함')
    elif pred_class_index == 3:
        print('hungry, 배고픔')
    elif pred_class_index == 4:
        print('sleepy, 졸림')
    else:
        print('모델예측잘못됨')
    return pred_class_name
    

def pipeline(audio_file_path):
    mfcc1 = process1(audio_file_path)
    pred1 = model1(mfcc1)
    
    mfcc2 = process2(audio_file_path)
    pred2 = model2(mfcc2)
    
    if pred1 == 0:
        print(pred2)
    else:
        print('당장 병원 ㄱㄱ')
    