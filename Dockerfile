# chatbot_api/Dockerfile

FROM python:3.9-slim

WORKDIR .

RUN pip install -r requirements.txt.

EXPOSE 8000
CMD ["sh", "entrypoint.sh"]