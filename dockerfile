FROM python:3.9-slim
COPY . .
RUN apt-get update && apt-get install -y --no-install-recommends locales git wget apt-utils  && \
    apt -y --no-install-recommends install make cmake gcc g++ build-essential && \  
    apt -y --no-install-recommends install libsndfile1-dev ffmpeg && \  
    pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
WORKDIR /app
CMD ["python3", "rag_en.py"]