FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y curl gnupg build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get update && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN npm install -g @angular/cli

WORKDIR /app

COPY . .

RUN npm install

EXPOSE 4201

CMD ["ng", "serve", "--host", "0.0.0.0", "--port", "4201", "--open", "--ssl", "true"]
