# Use the latest Ubuntu image
FROM ubuntu:latest

# Install prerequisites
RUN apt-get update && \
    apt-get install -y curl gnupg build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js (latest LTS version)
RUN curl -fsSL https://deb.nodesource.com/setup_current.x | bash - && \
    apt-get update && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Angular CLI
RUN npm install -g @angular/cli

# Create app root directory
WORKDIR /app

# Copy both apps
COPY ./pcx-admin-app /app/pcx-admin-app
COPY ./pcx-user-app /app/pcx-user-app

# Install dependencies for admin app
WORKDIR /app/pcx-admin-app
RUN npm install

# Install dependencies for user app
WORKDIR /app/pcx-user-app
RUN npm install

# Go back to root
WORKDIR /app

# Expose ports
EXPOSE 4200 4201

# Serve both apps in parallel
CMD bash -c "\
  cd /app/pcx-admin-app && ng serve --host 0.0.0.0 --port 4200 & \
  cd /app/pcx-user-app && ng serve --host 0.0.0.0 --port 4201 && \
  wait"
