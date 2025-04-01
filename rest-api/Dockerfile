FROM node:18

# Create app directory
WORKDIR /usr/src/app

# Install app dependencies
COPY package*.json ./
RUN npm install

# Copy all app source
COPY . .

# Optional: confirm permissions (not strictly required unless you hit permission issues)
# RUN chmod -R 755 /usr/src/app/ssl

# Expose HTTPS port (this is the internal port your app listens on, 8080)
EXPOSE 8080

# Start your app (HTTPS listener already in server.js)
CMD ["node", "server.js"]
