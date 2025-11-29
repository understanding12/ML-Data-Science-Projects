from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)


class RealTimeConversionPredictor:
    def __init__(self, model_path="data/lightgbm_conversion_pipeline.pkl"):
        try:
            self.model_data = joblib.load(model_path)
            self.predictor = self.model_data['pipeline']
            self.label_encoders = self.model_data['label_encoders']
            self.feature_names = self.model_data['feature_names']
            self.optimal_threshold = self.model_data.get('optimal_threshold', 0.5)
            self.target_action = set(['sub_car_claim_click', 'sub_car_claim_submit_click',
                                      'sub_open_dialog_click', 'sub_custom_question_submit_click',
                                      'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
                                      'sub_car_request_submit_click'])
            print("Предиктор для РЕАЛЬНОГО ВРЕМЕНИ загружен!")
            print(f"Задача: предсказать конверсию по текущему поведению ДО целевых действий")
            print(f"Порог: {self.optimal_threshold}")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            raise

    def create_single_user_features(self, session_data, hits_data=None):
        features = {
            'total_sessions': 1,
            'max_visit_number': int(session_data.get('visit_number', 1)),
            'most_common_source': session_data.get('utm_source', 'unknown'),
            'most_common_medium': session_data.get('utm_medium', 'unknown'),
            'most_common_campaign': session_data.get('utm_campaign', 'unknown'),
            'most_common_adcontent': session_data.get('utm_adcontent', 'unknown'),
            'most_common_device': session_data.get('device_category', 'unknown'),
            'most_common_brand': session_data.get('device_brand', 'unknown'),
            'most_common_browser': session_data.get('device_browser', 'unknown'),
            'most_common_country': session_data.get('geo_country', 'unknown'),
            'most_common_city': session_data.get('geo_city', 'unknown'),
            'first_visit': session_data.get('visit_date', '2023-01-01'),
            'last_visit': session_data.get('visit_date', '2023-01-01'),
            'unique_visit_days': 1,
            'most_common_time_of_day': 'afternoon',
            'avg_visit_hour': 12.0,
            'most_common_hour': 12,
            'has_converted': 0
        }

        if 'visit_time' in session_data:
            visit_time = str(session_data['visit_time'])
            if ':' in visit_time:
                try:
                    hour = int(visit_time.split(':')[0])
                    features['avg_visit_hour'] = float(hour)
                    features['most_common_hour'] = hour
                    if hour < 6:
                        features['most_common_time_of_day'] = 'night'
                    elif hour < 12:
                        features['most_common_time_of_day'] = 'morning'
                    elif hour < 18:
                        features['most_common_time_of_day'] = 'afternoon'
                    else:
                        features['most_common_time_of_day'] = 'evening'
                except:
                    pass

        if hits_data:
            hit_count = len(hits_data)
            features['hit_count'] = hit_count
            page_views = sum(1 for hit in hits_data if hit.get('hit_type') == 'PAGE')
            events = sum(1 for hit in hits_data if hit.get('hit_type') == 'EVENT')
            features['page_view_count'] = page_views
            features['event_count'] = events
            unique_actions = set(hit.get('event_action', '') for hit in hits_data)
            features['unique_actions_count'] = len(unique_actions)
            if hit_count > 1:
                features['approx_session_duration'] = hit_count * 10
            else:
                features['approx_session_duration'] = 0
            features['is_active_session'] = 1 if hit_count > 3 else 0
        return features

    def encode_features(self, features_dict):
        encoded_features = features_dict.copy()
        for feature_name, value in features_dict.items():
            if feature_name in self.label_encoders:
                try:
                    value_str = str(value)
                    if value_str in self.label_encoders[feature_name].classes_:
                        encoded_features[feature_name] = int(
                            self.label_encoders[feature_name].transform([value_str])[0])
                    else:
                        encoded_features[feature_name] = 0
                except Exception as e:
                    encoded_features[feature_name] = 0
        return encoded_features

    def create_feature_dataframe(self, encoded_features):
        feature_data = {}
        for feature_name in self.feature_names:
            if feature_name in encoded_features:
                value = encoded_features[feature_name]
                if isinstance(value, (np.integer, np.int64)):
                    feature_data[feature_name] = [int(value)]
                elif isinstance(value, (np.floating, np.float64)):
                    feature_data[feature_name] = [float(value)]
                else:
                    feature_data[feature_name] = [value]
            else:
                feature_data[feature_name] = [0]
        feature_df = pd.DataFrame(feature_data)
        feature_df = feature_df[self.feature_names]
        return feature_df

    def predict_conversion(self, session_data, hits_data=None):
        try:
            print(f"Анализируем визит {session_data.get('session_id', 'unknown')}")
            filtered_hits = []
            if hits_data:
                for hit in hits_data:
                    action = hit.get('event_action', '')
                    if action in self.target_action:
                        print(f"ВНИМАНИЕ: Обнаружено целевое действие '{action}'")
                        print("Для реального предсказания целевые действия должны отсутствовать")
                    filtered_hits.append(hit)
            print(f"Анализируем {len(filtered_hits)} хитов (текущее поведение)")
            features_dict = self.create_single_user_features(session_data, filtered_hits)
            encoded_features = self.encode_features(features_dict)
            feature_df = self.create_feature_dataframe(encoded_features)
            probability = self.predictor.predict_proba(feature_df)[0, 1]
            prediction = 1 if probability >= self.optimal_threshold else 0
            if prediction == 1:
                confidence = "ВЫСОКАЯ" if probability > 0.7 else "СРЕДНЯЯ" if probability > 0.3 else "НИЗКАЯ"
                recommendation = "Рекомендуется усилить взаимодействие"
            else:
                confidence = "НИЗКАЯ"
                recommendation = "Рекомендуется стимулировать активность"
            print(f"Предсказание: {prediction} (вероятность: {probability:.2%})")
            print(f"Уверенность: {confidence}")
            print(f"Рекомендация: {recommendation}")
            result = {
                'prediction': int(prediction),
                'probability': float(round(probability, 4)),
                'threshold': float(self.optimal_threshold),
                'confidence': confidence,
                'recommendation': recommendation,
                'hits_analyzed': len(filtered_hits),
                'client_id': str(session_data.get('client_id', 'unknown')),
                'session_id': str(session_data.get('session_id', 'unknown')),
                'success': True
            }
            return result
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return {
                'prediction': 0,
                'probability': 0.0,
                'threshold': float(self.optimal_threshold),
                'error': str(e),
                'success': False,
                'client_id': str(session_data.get('client_id', 'unknown')),
                'session_id': str(session_data.get('session_id', 'unknown'))
            }


try:
    predictor = RealTimeConversionPredictor()
    print("Модель для реального времени загружена!")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    predictor = None


@app.route('/')
def home():
    return jsonify({
        "message": "Real-time Conversion Prediction API",
        "status": "running",
        "model_loaded": predictor is not None,
        "purpose": "Predict conversion probability based on CURRENT user behavior (BEFORE target actions)"
    })


@app.route('/predict', methods=['POST'])
def predict_conversion():
    if predictor is None:
        return jsonify({
            "error": "Model not loaded",
            "success": False
        }), 500
    try:
        data = request.get_json()
        if not data or 'session_data' not in data:
            return jsonify({
                "error": "No session_data provided",
                "success": False
            }), 400
        session_data = data['session_data']
        hits_data = data.get('hits_data', [])
        required_fields = ['session_id', 'client_id', 'visit_date', 'visit_time']
        for field in required_fields:
            if field not in session_data:
                return jsonify({
                    "error": f"Missing required field: {field}",
                    "success": False
                }), 400
        result = predictor.predict_conversion(session_data, hits_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "success": False
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)