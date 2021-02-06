
FROM python:3.8

RUN curl https://download.pytorch.org/models/vgg19-dcbb9e9d.pth -o vgg19-dcbb9e9d.pth

ENV MODEL_PATH='vgg19-dcbb9e9d.pth'

COPY requirements.txt .

RUN pip install -r /requirements.txt

COPY . .

EXPOSE 8501

CMD streamlit run app.py --server.port $PORT