FROM python:3.11-slim

WORKDIR /app/

COPY . /app/

RUN pip install -r requirements.txt

EXPOSE 8001

CMD ["uvicorn","server.server:app","--host","0.0.0.0","--port","8001"]

