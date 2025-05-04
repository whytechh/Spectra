FROM pytorch/pytorch:latest

COPY /app/requirements.txt code/requirements.txt

RUN pip install -r code/requirements.txt

COPY app/ code/app/
COPY model_selector.py code/app/model_selector.py
COPY labels.json code/app/labels.json

WORKDIR code/app
CMD python main.py

