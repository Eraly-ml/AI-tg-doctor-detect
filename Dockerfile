# Используйте базовый образ
FROM python:3.10-slim

# Установите рабочую директорию
WORKDIR /app

# Копируйте файлы модели в контейнер
COPY brain_model.pkl /app/models/brain_model.pkl
COPY eye_model.pkl /app/models/eye_model.pkl

# Установите зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируйте остальной код
COPY . .

# Определите команду для запуска контейнера
CMD ["python", "bot.py"]

