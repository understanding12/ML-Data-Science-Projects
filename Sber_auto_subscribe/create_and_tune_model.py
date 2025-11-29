import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


class CustomerConversionPredictor:
    def __init__(self):
        self.pipeline = None
        self.label_encoders = {}
        self.categorical_columns = None
        self.numerical_columns = None
        self.feature_names = None
        self.best_params = None
        self.optimal_threshold = 0.5
        self.target_action = set(['sub_car_claim_click', 'sub_car_claim_submit_click',
                                  'sub_open_dialog_click', 'sub_custom_question_submit_click',
                                  'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
                                  'sub_car_request_submit_click'])

    def clear_memory(self):
        gc.collect()
        print("Память очищена")

    def check_and_generate_features(self, hits_path="ga_hits.csv", sessions_path="ga_sessions.csv", force_recreate=False):
        print("Запускаю процесс подготовки данных...")
        print("=" * 50)
        os.makedirs('data', exist_ok=True)
        merged_file = 'data/df_merged.csv'
        if os.path.exists(merged_file) and not force_recreate:
            print("Загружаю объединенные данные...")
        else:
            print("Создаю объединенные данные...")
            self.load_and_save_merge_data(hits_path, sessions_path)
            self.clear_memory()
        result_file = 'data/result_df.csv'
        if os.path.exists(result_file) and not force_recreate:
            print("Загружаю обработанные данные...")
        else:
            print("Создаю обработанные данные...")
            self.load_merge_data(result_file, sessions_path)
            self.clear_memory()
        features_file = 'data/user_features.csv'
        if os.path.exists(features_file) and not force_recreate:
            print("Загружаю пользовательские признаки...")
        else:
            print("Создаю пользовательские признаки...")
            self.load_result_data(result_file)
            self.clear_memory()
        print("Загружаю финальные данные для обучения...")
        X, y = self.load_data_for_train()
        print("=" * 50)
        print("Все данные готовы для обучения модели!")
        print(f"X: {X.shape}, y: {y.shape}")
        print(f"Баланс классов: {y.value_counts().to_dict()}")
        return X, y

    def load_and_save_merge_data(self, hits_path, sessions_path):
        print("Загружаю и объединяем данные...")
        data = pd.read_csv("data/" + hits_path)
        data_s = pd.read_csv("data/" + sessions_path)
        data["event_action"] = data["event_action"].apply(lambda x: 1 if x in self.target_action else 0)
        session_sums = data.groupby('session_id')['event_action'].sum().reset_index()
        session_sums['has_events'] = (session_sums['event_action'] > 0).astype(int)
        df_merged = data.merge(session_sums[['session_id', 'has_events']], on='session_id')
        df_merged.to_csv("data/df_merged.csv", index=False)
        print(f"Объединенные данные сохранены: {df_merged.shape}")

    def load_merge_data(self, result_path, sessions_path):
        print("Обрабатываем данные...")
        data_s = pd.read_csv("data/" + sessions_path)
        df_merged = pd.read_csv("data/df_merged.csv")
        result_df = data_s.merge(df_merged, on='session_id')
        result_df = result_df.drop(
            ['utm_keyword', 'device_os', 'device_model', 'hit_time', 'hit_referer', 'event_value', 'hit_page_path',
             'event_category', 'hit_date', 'hit_number', 'hit_type', 'hit_page_path', 'event_label', 'event_action'],
            axis=1)
        result_df.dropna(inplace=True)
        result_df['visit_hour'] = result_df['visit_time'].str[:2].astype(int)
        result_df['time_of_day'] = result_df['visit_hour'].apply(lambda h: 'night' if h < 6 else
        'morning' if h < 12 else
        'afternoon' if h < 18 else 'evening')
        result_df.to_csv(result_path, index=False)
        print(f"Обработанные данные сохранены: {result_df.shape}")

    def load_result_data(self, result_path):
        print("Создаю пользовательские признаки...")
        result_df = pd.read_csv(result_path)
        user_features = result_df.groupby('client_id').agg(
            total_sessions=('session_id', 'nunique'),
            max_visit_number=('visit_number', 'max'),
            most_common_source=('utm_source', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'),
            most_common_medium=('utm_medium', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'),
            most_common_campaign=('utm_campaign', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'),
            most_common_adcontent=('utm_adcontent', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'),
            most_common_device=('device_category', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'),
            most_common_brand=('device_brand', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'),
            most_common_browser=('device_browser', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'),
            most_common_country=('geo_country', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'),
            most_common_city=('geo_city', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'),
            has_converted=('has_events', 'max'),
            first_visit=('visit_date', 'min'),
            last_visit=('visit_date', 'max'),
            unique_visit_days=('visit_date', 'nunique'),
            most_common_time_of_day=('time_of_day', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'),
            avg_visit_hour=('visit_hour', 'mean'),
            most_common_hour=('visit_hour', lambda x: x.mode()[0] if len(x.mode()) > 0 else 12)
        ).reset_index()

        user_features = user_features.drop('client_id', axis=1)
        user_features = user_features.drop_duplicates()
        numeric_columns = user_features.select_dtypes(include=[np.number]).columns
        user_features[numeric_columns] = user_features[numeric_columns].fillna(0)
        categorical_columns = user_features.select_dtypes(include=['object']).columns
        user_features[categorical_columns] = user_features[categorical_columns].fillna('unknown')
        user_features.to_csv('data/user_features.csv', index=False)
        print(f"Пользовательские признаки сохранены: {user_features.shape}")

    def load_data_for_train(self):
        print("Подготавливаем данные для обучения...")
        user_features = pd.read_csv('data/user_features.csv')
        self.categorical_columns = user_features.select_dtypes(include=['object']).columns.tolist()
        self.numerical_columns = user_features.select_dtypes(include=[np.number]).columns.tolist()
        if 'has_converted' in self.numerical_columns:
            self.numerical_columns.remove('has_converted')
        print(f"Категориальные признаки: {len(self.categorical_columns)}")
        print(f"Числовые признаки: {len(self.numerical_columns)}")
        X_encoded = user_features.copy()
        for col in self.categorical_columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            self.label_encoders[col] = le
        X = X_encoded.drop('has_converted', axis=1)
        y = X_encoded['has_converted']
        self.feature_names = X.columns.tolist()
        print(f"Данные для обучения готовы: X {X.shape}, y {y.shape}")
        return X, y

    def create_pipeline(self, use_tuned_params=True):
        print("Создаем пайплайн...")

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_columns),
                ('cat', 'passthrough', self.categorical_columns)
            ]
        )

        if use_tuned_params and self.best_params:
            lgb_params = self.best_params
            print("Использую настроенные параметры")
        else:
            lgb_params = {
                'n_estimators': 100,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'class_weight': 'balanced',
                'random_state': 42,
                'verbose': -1
            }
            print("Использую параметры по умолчанию")

        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LGBMClassifier(**lgb_params))
        ])

        print("Пайплайн создан")

    def tune_hyperparameters(self, X_train, y_train):
        print("Начинаем подбор гиперпараметров для LightGBM...")

        lgb_params = {
            'classifier__n_estimators': [100, 200, 300, 500],
            'classifier__max_depth': [6, 8, 10, 12, 15],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15],
            'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'classifier__reg_alpha': [0, 0.1, 0.5, 1.0],
            'classifier__reg_lambda': [0, 0.1, 0.5, 1.0]
        }

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_columns),
                ('cat', 'passthrough', self.categorical_columns)
            ]
        )

        base_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LGBMClassifier(
                class_weight='balanced',
                random_state=42,
                verbose=-1
            ))
        ])

        lgb_search = RandomizedSearchCV(
            base_pipeline,
            lgb_params,
            n_iter=15,
            scoring='roc_auc',
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        print("Запускаю поиск гиперпараметров...")
        lgb_search.fit(X_train, y_train)
        self.best_params = {}
        for key, value in lgb_search.best_params_.items():
            param_name = key.replace('classifier__', '')
            self.best_params[param_name] = value

        print(f"Лучшие параметры: {self.best_params}")
        print(f"Лучший ROC-AUC на кросс-валидации: {lgb_search.best_score_:.4f}")

        return lgb_search.best_score_

    def train_model(self, X, y, test_size=0.2, random_state=42, tune_hyperparams=True):
        print("Начинаю обучение модели...")
        print("=" * 50)

        print("Разделяю данные на train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Обучающая выборка: {X_train.shape}")
        print(f"Тестовая выборка: {X_test.shape}")

        if tune_hyperparams:
            self.tune_hyperparameters(X_train, y_train)

        print("Создаю финальный пайплайн...")
        self.create_pipeline(use_tuned_params=tune_hyperparams)

        print("🎓 Обучаю модель...")
        self.pipeline.fit(X_train, y_train)

        print("Оцениваю модель...")
        self.evaluate_model(X_test, y_test)

        return X_train, X_test, y_train, y_test

    def evaluate_model(self, X_test, y_test):
        print("\n Оцениваем модель...")

        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]

        print("\n" + "=" * 50)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
        print("=" * 50)
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print("\n Classification Report:")
        print(classification_report(y_test, y_pred))

        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Conversion', 'Conversion'],
                    yticklabels=['No Conversion', 'Conversion'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

        if hasattr(self.pipeline.named_steps['classifier'], 'feature_importances_'):
            feature_importances = self.pipeline.named_steps['classifier'].feature_importances_
            feature_names = self.feature_names
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importances
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df.head(15), x='importance', y='feature')
            plt.title('Top 15 Feature Importances (LightGBM)')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()

            print("\n Топ-10 самых важных признаков:")
            for i, row in importance_df.head(10).iterrows():
                print(f"  {i + 1:2d}. {row['feature']}: {row['importance']:.4f}")

    def set_threshold(self, threshold):
        self.optimal_threshold = threshold
        print(f" Порог установлен: {self.optimal_threshold}")

    def predict(self, new_data, use_optimal_threshold=True):
        if self.pipeline is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")

        if use_optimal_threshold and self.optimal_threshold != 0.5:
            probabilities = self.predict_proba(new_data)
            return (probabilities[:, 1] >= self.optimal_threshold).astype(int)
        else:
            return self.pipeline.predict(new_data)

    def find_and_set_optimal_threshold(self, X_test, y_test, method='f1'):
        from sklearn.metrics import precision_recall_curve, f1_score
        y_proba = self.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = f1_scores[:-1]

        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]

        self.set_threshold(optimal_threshold)

        print(f" Найден оптимальный порог: {optimal_threshold:.4f}")
        print(f" F1-score при этом пороге: {f1_scores[optimal_idx]:.4f}")
        return optimal_threshold

    def predict_proba(self, new_data):
        if self.pipeline is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")

        return self.pipeline.predict_proba(new_data)

    def save_model(self, filepath="data/lightgbm_conversion_pipeline.pkl"):
        if self.pipeline is None:
            raise ValueError("Модель не обучена. Нечего сохранять.")

        model_data = {
            'pipeline': self.pipeline,
            'label_encoders': self.label_encoders,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'optimal_threshold': self.optimal_threshold
        }

        joblib.dump(model_data, filepath)
        print(f" Модель сохранена в {filepath}")
        print(f" Оптимальный порог также сохранен: {self.optimal_threshold}")

    def load_model(self, filepath="data/lightgbm_conversion_pipeline.pkl"):
        model_data = joblib.load(filepath)

        self.pipeline = model_data['pipeline']
        self.label_encoders = model_data['label_encoders']
        self.categorical_columns = model_data['categorical_columns']
        self.numerical_columns = model_data['numerical_columns']
        self.feature_names = model_data['feature_names']
        self.best_params = model_data.get('best_params', {})
        self.optimal_threshold = model_data.get('optimal_threshold', 0.5)

        print(f"📂 Модель загружена из {filepath}")
        print(f"🎯 Оптимальный порог: {self.optimal_threshold}")

    def full_pipeline(self, hits_path="ga_hits.csv", sessions_path="ga_sessions.csv",
                      tune_hyperparams=True, force_recreate=False, find_optimal_threshold=True):
        print(" ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА")
        print("=" * 60)

        X, y = self.check_and_generate_features(
            hits_path=hits_path,
            sessions_path=sessions_path,
            force_recreate=force_recreate
        )

        X_train, X_test, y_train, y_test = self.train_model(
            X, y,
            tune_hyperparams=tune_hyperparams
        )

        if find_optimal_threshold:
            print("\n ПОДБИРАЮ ОПТИМАЛЬНЫЙ ПОРОГ КЛАССИФИКАЦИИ...")
            optimal_threshold = self.find_and_set_optimal_threshold(X_test, y_test)

            print("\n ОЦЕНКА С ОПТИМАЛЬНЫМ ПОРОГОМ:")
            y_pred_optimal = self.predict(X_test, use_optimal_threshold=True)

            from sklearn.metrics import classification_report
            print(classification_report(y_test, y_pred_optimal))

        self.save_model()

        print("=" * 60)
        print(" ПОЛНЫЙ ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!")
        print(f" Оптимальный порог: {self.optimal_threshold:.4f}")

        return X_test, y_test


def train_full_model(hits_path="ga_hits.csv", sessions_path="ga_sessions.csv",
                     tune_hyperparams=True, force_recreate=False, find_optimal_threshold=True):
    predictor = CustomerConversionPredictor()
    X_test, y_test = predictor.full_pipeline(
        hits_path=hits_path,
        sessions_path=sessions_path,
        tune_hyperparams=tune_hyperparams,
        force_recreate=force_recreate,
        find_optimal_threshold=find_optimal_threshold
    )
    return predictor, X_test, y_test


def load_trained_model(filepath="data/lightgbm_conversion_pipeline.pkl"):
    predictor = CustomerConversionPredictor()
    predictor.load_model(filepath)
    return predictor


if __name__ == "__main__":
    predictor, X_test, y_test = train_full_model(
        hits_path="ga_hits.csv",
        sessions_path="ga_sessions.csv",
        tune_hyperparams=True,
        force_recreate=False,
        find_optimal_threshold=True
    )