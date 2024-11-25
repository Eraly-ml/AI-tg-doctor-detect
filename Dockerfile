# Используйте базовый образ
FROM python:3.10-slim

# Установите рабочую директорию
WORKDIR /app

# Копируйте файлы модели в контейнер
COPY brain_model.pkl /app/brain_model.pkl
COPY eye_model.pkl /app/eye_model.pkl

# Установите зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir scipy


# Копируйте остальной код
COPY . .

# Определите команду для запуска контейнера
CMD ["python", "bot.py"]

