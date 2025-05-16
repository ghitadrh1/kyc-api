FROM python:3.10-bullseye

RUN apt-get update && apt-get install -y \
    git build-essential \
    libleptonica-dev libtesseract-dev \
    libjpeg62-turbo-dev libpng-dev libtiff5-dev zlib1g-dev \
    autoconf automake libtool pkg-config \
    poppler-utils tesseract-ocr-fra tesseract-ocr-ara \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch 5.3.0 https://github.com/tesseract-ocr/tesseract.git && \
    cd tesseract && \
    ./autogen.sh && \
    ./configure && \
    make && \
    make install && \
    ldconfig && \
    cd .. && rm -rf tesseract

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
