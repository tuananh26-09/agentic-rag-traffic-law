FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Chạy trang Web khi Container bật lên
CMD ["streamlit", "run", "web_app.py", "--server.port=8501", "--server.address=0.0.0.0"]