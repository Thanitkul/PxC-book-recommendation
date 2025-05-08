# Use official Node.js image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Install Angular CLI globally
RUN npm install -g @angular/cli

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the application code
COPY . .

# Expose application port
EXPOSE 4200

# Start the Angular application
CMD ["ng", "serve", "--host", "0.0.0.0", "--port", "4200", "--open", "--ssl", "true"]
