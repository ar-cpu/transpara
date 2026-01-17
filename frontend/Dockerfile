FROM node:21-alpine

WORKDIR /app

# Install dependencies
COPY package.json .
RUN npm install

# Copy source code
COPY . .

# Expose port
EXPOSE 4200

# Start development server
CMD ["npm", "start", "--", "--host", "0.0.0.0", "--disable-host-check"]
