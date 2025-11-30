# АналитикаКонверсийAI

## Немного о задаче:

```bash
git clone https://github.com/understanding12/ML-Data-Science-Projects.git
cd ML-Data-Science-Projects/Sber_auto_subscribe
# Создать папку для данных
mkdir -p data

# Скачать данные по ссылке:
# https://drive.google.com/drive/folders/1rA4o6KHH-M2KMvBLHp5DZ5gioF2q7hZw?usp=sharing

# Распаковать файлы в папку data/:
# - ga_sessions.csv
# - ga_hits.csv
# Сборка и запуск контейнера
docker-compose up --build

# Или в фоновом режиме
docker-compose up -d --build
# Проверить API
curl http://localhost:5000

# Запустить тесты
docker-compose exec ml-api python test_api.py

# Запуск обучения в контейнере
docker-compose exec ml-api python create_and_tune_model.py
