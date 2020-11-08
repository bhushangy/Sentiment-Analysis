FROM python:3.7

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm

EXPOSE 5001

CMD [ "python", "DepressionAnalysisApp.py" ]