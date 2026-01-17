import os
import re
import hashlib
import magic
from functools import wraps
from flask import request, jsonify
from werkzeug.utils import secure_filename


def setup_security_headers(app):

    @app.after_request
    def add_security_headers(response):
        # prevent xss attacks
        response.headers['X-XSS-Protection'] = '1; mode=block'

        # prevent mime type sniffing
        response.headers['X-Content-Type-Options'] = 'nosniff'

        # prevent clickjacking
        response.headers['X-Frame-Options'] = 'DENY'

        # content security policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
            "img-src 'self' data: https:; "
            "font-src 'self' https://cdnjs.cloudflare.com; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )
        response.headers['Content-Security-Policy'] = csp

        # enforce https in production
        if not app.config['DEBUG']:
            hsts_value = f"max-age={app.config['HSTS_MAX_AGE']}"
            if app.config['HSTS_INCLUDE_SUBDOMAINS']:
                hsts_value += "; includeSubDomains"
            response.headers['Strict-Transport-Security'] = hsts_value

        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'

        return response


def sanitize_filename(filename):
    # prevent directory traversal and command injection
    filename = secure_filename(filename)
    filename = re.sub(r'[^\w\s.-]', '', filename)

    max_length = 255
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext

    return filename


def validate_file_extension(filename, allowed_extensions):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in allowed_extensions


def validate_file_magic_number(file_path, allowed_mime_types):
    # check actual file type using magic numbers, not just extension
    try:
        mime = magic.Magic(mime=True)
        file_mime_type = mime.from_file(file_path)
        return file_mime_type in allowed_mime_types
    except Exception:
        return False


def validate_file_size(file_path, max_size_bytes):
    try:
        file_size = os.path.getsize(file_path)
        return file_size <= max_size_bytes
    except Exception:
        return False


def sanitize_input(text, max_length=100000):
    # prevent xss and injection attacks
    if not text:
        return ""

    text = text.replace('\x00', '')
    text = text[:max_length]

    # remove dangerous patterns
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
    ]

    for pattern in dangerous_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    return text


def sanitize_sql_input(text):
    # extra protection against sql injection (use parameterized queries too)
    if not text:
        return ""

    text = text.replace('--', '').replace('/*', '').replace('*/', '')
    text = text.replace(';', '')

    return text


def generate_secure_token(length=32):
    return hashlib.sha256(os.urandom(length)).hexdigest()


def validate_request_size(max_size_mb=16):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            content_length = request.content_length
            if content_length and content_length > max_size_mb * 1024 * 1024:
                return jsonify({
                    'error': 'Request too large',
                    'message': f'Maximum request size is {max_size_mb}MB'
                }), 413
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def validate_content_type(allowed_types):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.content_type not in allowed_types:
                return jsonify({
                    'error': 'Invalid content type',
                    'message': f'Content type must be one of: {", ".join(allowed_types)}'
                }), 400
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def secure_delete_file(file_path, passes=3):
    # DoD 5220.22-M standard: overwrite file before deletion
    try:
        if not os.path.exists(file_path):
            return

        file_size = os.path.getsize(file_path)

        with open(file_path, 'ba+') as f:
            for _ in range(passes):
                f.seek(0)
                f.write(os.urandom(file_size))

        os.remove(file_path)
    except Exception:
        if os.path.exists(file_path):
            os.remove(file_path)


def check_sql_injection(text):
    sql_keywords = [
        'union', 'select', 'insert', 'update', 'delete', 'drop',
        'create', 'alter', 'exec', 'execute', 'script', 'javascript'
    ]

    text_lower = text.lower()
    for keyword in sql_keywords:
        if keyword in text_lower:
            return True
    return False


def check_command_injection(text):
    dangerous_chars = ['|', ';', '&', '$', '`', '\n', '>', '<', '(', ')', '{', '}']
    return any(char in text for char in dangerous_chars)


def check_path_traversal(path):
    dangerous_patterns = ['../', '..\\', '%2e%2e', '..%2f', '..%5c']
    path_lower = path.lower()
    return any(pattern in path_lower for pattern in dangerous_patterns)


class RateLimitExceeded(Exception):
    pass


class SecurityViolation(Exception):
    pass
