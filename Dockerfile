FROM python:3.9.6

EXPOSE 8501

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["streamlit","run","main.py"]