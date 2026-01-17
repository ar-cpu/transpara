// Production environment configuration
export const environment = {
  production: true,
  apiUrl: 'http://localhost:5000/api/v1',
  apiTimeout: 120000, // 2 minutes
  maxRetries: 3,
  retryDelay: 2000, // 2 seconds
  enableLogging: false,
  enableOfflineDetection: true
};
