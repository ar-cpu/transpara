# SSL Certificates

## Production Deployment

For production, you should use Let's Encrypt to obtain free SSL certificates:

```bash
# Install Certbot
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d transpara.com -d www.transpara.com

# Auto-renewal (certbot handles this automatically)
sudo certbot renew --dry-run
```

## Development/Testing

For development, you can generate self-signed certificates:

```bash
# Generate self-signed certificate (valid for 365 days)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout transpara.key \
  -out transpara.crt \
  -subj "/C=US/ST=State/L=City/O=Transpara/CN=localhost"
```

## Files Required

- `transpara.crt` - SSL certificate
- `transpara.key` - Private key
- `dhparam.pem` - (Optional) Diffie-Hellman parameters for enhanced security

## Security Best Practices

1. Keep private keys secure (chmod 600)
2. Use strong ciphers (TLSv1.2+ only)
3. Enable HSTS
4. Implement certificate pinning for mobile apps
5. Monitor certificate expiration dates
6. Use OCSP stapling
