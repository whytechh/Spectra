FROM python:3.9

COPY ./requirements.txt code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY app/ code/app/

CMD python code/app/main.py

