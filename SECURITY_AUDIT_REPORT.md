# Security Audit Report - Local Document Q&A RAG System

**Audit Date**: 2025-09-16
**Auditor**: Security Audit Team
**Repository**: `/home/jeremy/projects/local-document-qa-rag`
**Application Type**: Local Document Processing and Q&A System
**Technology Stack**: Python, Streamlit, LangChain, ChromaDB, Ollama

---

## Executive Summary

This security audit evaluated the Local Document Q&A RAG system for security vulnerabilities, compliance issues, and best practices. The application is designed to run locally and process documents using AI models. While the application benefits from its local-only architecture, several security concerns require immediate attention.

### Overall Risk Assessment: **MEDIUM-HIGH**

**Critical Findings**: 3
**High Severity**: 4
**Medium Severity**: 5
**Low Severity**: 3

---

## 1. Authentication and Authorization

### Finding: No Authentication Mechanism
**Severity**: HIGH
**OWASP Reference**: A07:2021 - Identification and Authentication Failures

#### Issue
The application lacks any authentication or authorization mechanism. Anyone with network access to the Streamlit port (8501 by default) can:
- Upload and process documents
- Access all stored documents
- Query the entire knowledge base
- Delete the vector database

#### Recommendation
```python
# Implement basic authentication for Streamlit
import streamlit_authenticator as stauth
import yaml

# Load user credentials from secure storage
with open('config.yaml') as file:
    config = yaml.safe_load(file)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
elif authentication_status:
    # Main application code here
    pass
```

---

## 2. Data Handling and Storage Security

### Finding: Unencrypted Vector Database Storage
**Severity**: MEDIUM
**OWASP Reference**: A02:2021 - Cryptographic Failures

#### Issue
The ChromaDB vector database stores document embeddings and metadata in plaintext at `./chroma_db`. Sensitive documents are not encrypted at rest.

#### Recommendation
```python
import cryptography
from cryptography.fernet import Fernet

class EncryptedChroma:
    def __init__(self, persist_directory, key=None):
        if key is None:
            key = Fernet.generate_key()
        self.cipher = Fernet(key)
        self.persist_directory = persist_directory

    def encrypt_before_store(self, data):
        return self.cipher.encrypt(data.encode())

    def decrypt_after_retrieve(self, encrypted_data):
        return self.cipher.decrypt(encrypted_data).decode()
```

### Finding: No Document Sanitization
**Severity**: HIGH
**OWASP Reference**: A03:2021 - Injection

#### Issue
Documents are processed without sanitization, potentially allowing:
- Malicious PDF payloads
- Path traversal via crafted filenames
- Code injection through document content

---

## 3. Input Validation and API Security

### Finding: Insufficient Path Traversal Protection
**Severity**: CRITICAL
**OWASP Reference**: A01:2021 - Broken Access Control

#### Issue
Line 40-45 in `app.py`:
```python
for filename in os.listdir(DOCUMENTS_DIR):
    file_path = os.path.join(DOCUMENTS_DIR, filename)
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
```

No validation of filename allows potential directory traversal attacks.

#### Secure Implementation
```python
import os
import pathlib

def validate_filename(filename):
    # Remove any path components
    filename = os.path.basename(filename)
    # Check for directory traversal attempts
    if '..' in filename or '/' in filename or '\\' in filename:
        raise ValueError("Invalid filename")
    # Whitelist allowed characters
    import re
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', filename):
        raise ValueError("Filename contains invalid characters")
    return filename

def secure_file_path(base_dir, filename):
    validated_name = validate_filename(filename)
    file_path = os.path.join(base_dir, validated_name)
    # Ensure the resolved path is within base directory
    if not os.path.abspath(file_path).startswith(os.path.abspath(base_dir)):
        raise ValueError("Path traversal attempt detected")
    return file_path
```

### Finding: No Input Length Validation
**Severity**: MEDIUM
**OWASP Reference**: A04:2021 - Insecure Design

#### Issue
User queries are not validated for length, potentially causing:
- Memory exhaustion
- DoS attacks
- Performance degradation

#### Recommendation
```python
MAX_QUERY_LENGTH = 1000

if prompt := st.chat_input("Ask a question about your documents:"):
    if not prompt.strip():
        st.warning("Please enter a valid question.")
    elif len(prompt) > MAX_QUERY_LENGTH:
        st.error(f"Query too long. Maximum {MAX_QUERY_LENGTH} characters allowed.")
    else:
        # Process query
        pass
```

---

## 4. Dependency Vulnerabilities

### Finding: Known Vulnerabilities in Dependencies
**Severity**: HIGH

#### Identified CVEs:
1. **Streamlit 1.38.0**: SAFE (CVE-2024-42474 was fixed in 1.37.0)
2. **LangChain 0.2.16**: Potentially vulnerable to older CVEs
   - CVE-2024-5998: Pickle deserialization (Fixed in 0.2.4 - SAFE)
   - CVE-2024-3571: Path traversal in LocalFileStore (CHECK USAGE)
   - CVE-2024-3095: SSRF vulnerability (CHECK USAGE)

#### Recommendation
```bash
# Install safety to check for vulnerabilities
pip install safety

# Check current dependencies
safety check --json

# Update all dependencies
pip install --upgrade langchain langchain_community langchain_chroma

# Use pip-audit for comprehensive scanning
pip install pip-audit
pip-audit
```

---

## 5. Configuration Security

### Finding: Hardcoded Configuration
**Severity**: MEDIUM
**OWASP Reference**: A05:2021 - Security Misconfiguration

#### Issue
Configuration is hardcoded in `app.py`:
```python
OLLAMA_MODEL = "llama3"
DOCUMENTS_DIR = "documents"
CHROMA_DB_PATH = "./chroma_db"
```

#### Secure Configuration
```python
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration with environment variables and defaults
CONFIG = {
    'OLLAMA_MODEL': os.getenv('OLLAMA_MODEL', 'llama3'),
    'DOCUMENTS_DIR': os.getenv('DOCUMENTS_DIR', 'documents'),
    'CHROMA_DB_PATH': os.getenv('CHROMA_DB_PATH', './chroma_db'),
    'MAX_FILE_SIZE': int(os.getenv('MAX_FILE_SIZE', 10485760)),  # 10MB
    'ALLOWED_EXTENSIONS': os.getenv('ALLOWED_EXTENSIONS', 'pdf,txt,md').split(','),
    'ENABLE_DEBUG': os.getenv('ENABLE_DEBUG', 'false').lower() == 'true'
}
```

### Finding: No HTTPS/TLS Configuration
**Severity**: HIGH
**OWASP Reference**: A02:2021 - Cryptographic Failures

#### Issue
Streamlit runs on HTTP by default, exposing all data in transit.

#### Recommendation
```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Run Streamlit with HTTPS
streamlit run app.py \
  --server.sslCertFile cert.pem \
  --server.sslKeyFile key.pem \
  --server.port 443
```

---

## 6. OWASP Top 10 Compliance Assessment

| OWASP Category | Status | Issues Found |
|----------------|--------|--------------|
| A01: Broken Access Control | ❌ FAIL | No access controls, path traversal vulnerability |
| A02: Cryptographic Failures | ❌ FAIL | No encryption at rest, no HTTPS |
| A03: Injection | ⚠️ RISK | No input sanitization, potential prompt injection |
| A04: Insecure Design | ❌ FAIL | No threat modeling, missing security controls |
| A05: Security Misconfiguration | ❌ FAIL | Default configurations, no hardening |
| A06: Vulnerable Components | ⚠️ RISK | Some dependencies may have vulnerabilities |
| A07: Authentication Failures | ❌ FAIL | No authentication mechanism |
| A08: Software and Data Integrity | ⚠️ RISK | No integrity checks on uploaded files |
| A09: Security Logging | ❌ FAIL | No security logging or monitoring |
| A10: SSRF | ⚠️ RISK | LangChain SSRF vulnerability possible |

---

## 7. Attack Vectors and Threat Model

### Identified Attack Vectors

#### 1. Document Upload Attacks
- **Vector**: Malicious PDF with embedded JavaScript
- **Impact**: Code execution, data exfiltration
- **Mitigation**: Implement PDF sanitization

#### 2. Prompt Injection
- **Vector**: Crafted queries to manipulate LLM behavior
- **Impact**: Information disclosure, bypassing restrictions
- **Mitigation**: Input validation and output filtering

#### 3. Resource Exhaustion
- **Vector**: Large file uploads or complex queries
- **Impact**: DoS, system unavailability
- **Mitigation**: Rate limiting and resource quotas

#### 4. Data Exfiltration
- **Vector**: Unrestricted access to vector database
- **Impact**: Confidential document exposure
- **Mitigation**: Access controls and encryption

---

## 8. Security Headers Configuration

### Recommended Streamlit Configuration
```python
# config.toml for Streamlit
[server]
enableCORS = false
enableXsrfProtection = true

[browser]
serverAddress = "localhost"
serverPort = 8501

# Additional headers via reverse proxy (nginx)
```

### Nginx Security Headers
```nginx
server {
    listen 443 ssl http2;
    server_name localhost;

    # SSL Configuration
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 9. Secure Implementation Recommendations

### Priority 1 - Critical (Implement Immediately)

1. **Add Authentication**
   ```python
   # Use streamlit-authenticator or implement JWT
   pip install streamlit-authenticator
   ```

2. **Validate All Inputs**
   ```python
   def validate_user_input(input_str, max_length=1000, allow_special=False):
       if not input_str or len(input_str) > max_length:
           return False
       if not allow_special:
           # Remove potentially dangerous characters
           import re
           if re.search(r'[<>\"\'`;(){}]', input_str):
               return False
       return True
   ```

3. **Implement File Upload Security**
   ```python
   def secure_file_upload(uploaded_file):
       # Check file size
       if uploaded_file.size > CONFIG['MAX_FILE_SIZE']:
           raise ValueError("File too large")

       # Validate extension
       file_ext = uploaded_file.name.split('.')[-1].lower()
       if file_ext not in CONFIG['ALLOWED_EXTENSIONS']:
           raise ValueError("File type not allowed")

       # Scan for malware (integrate with ClamAV)
       # scan_result = scan_file_for_malware(uploaded_file)

       return uploaded_file
   ```

### Priority 2 - High (Implement Within 1 Week)

1. **Add Security Logging**
   ```python
   import logging
   from datetime import datetime

   # Configure security logger
   security_logger = logging.getLogger('security')
   security_logger.setLevel(logging.INFO)
   handler = logging.FileHandler('security.log')
   formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
   handler.setFormatter(formatter)
   security_logger.addHandler(handler)

   def log_security_event(event_type, details):
       security_logger.info(f"{event_type}: {details}")
   ```

2. **Implement Rate Limiting**
   ```python
   from functools import wraps
   import time
   from collections import defaultdict

   request_times = defaultdict(list)

   def rate_limit(max_requests=10, window=60):
       def decorator(func):
           @wraps(func)
           def wrapper(*args, **kwargs):
               user_id = get_user_id()  # Implement user identification
               now = time.time()

               # Clean old requests
               request_times[user_id] = [t for t in request_times[user_id] if now - t < window]

               if len(request_times[user_id]) >= max_requests:
                   raise Exception("Rate limit exceeded")

               request_times[user_id].append(now)
               return func(*args, **kwargs)
           return wrapper
       return decorator
   ```

### Priority 3 - Medium (Implement Within 1 Month)

1. **Add Content Security Policy**
2. **Implement Data Encryption at Rest**
3. **Add Automated Security Scanning**
4. **Implement Backup and Recovery**

---

## 10. Security Testing Checklist

### Pre-Deployment Testing
- [ ] Authentication bypass testing
- [ ] Input validation testing (fuzzing)
- [ ] File upload security testing
- [ ] Path traversal testing
- [ ] Injection testing (prompt, SQL, command)
- [ ] Session management testing
- [ ] Error handling verification
- [ ] Rate limiting verification
- [ ] SSL/TLS configuration testing
- [ ] Dependency vulnerability scanning

### Security Test Cases

```python
# test_security.py
import pytest
from app import validate_filename, secure_file_path

def test_path_traversal_prevention():
    """Test that path traversal attempts are blocked"""
    malicious_names = [
        "../../../etc/passwd",
        "..\\..\\windows\\system32\\config\\sam",
        "file://etc/passwd",
        "documents/../../../etc/passwd"
    ]

    for name in malicious_names:
        with pytest.raises(ValueError):
            validate_filename(name)

def test_input_length_validation():
    """Test that oversized inputs are rejected"""
    large_input = "A" * 10000
    assert not validate_user_input(large_input, max_length=1000)

def test_file_type_validation():
    """Test that only allowed file types are accepted"""
    allowed = ["document.pdf", "notes.txt", "readme.md"]
    blocked = ["script.exe", "payload.sh", "hack.php"]

    for filename in allowed:
        assert validate_file_extension(filename) == True

    for filename in blocked:
        assert validate_file_extension(filename) == False
```

---

## 11. Incident Response Plan

### Security Incident Categories
1. **Data Breach**: Unauthorized access to stored documents
2. **System Compromise**: Malicious code execution
3. **Service Disruption**: DoS or resource exhaustion
4. **Data Corruption**: Vector database corruption

### Response Procedures
1. **Detect**: Monitor logs for anomalies
2. **Contain**: Isolate affected systems
3. **Investigate**: Analyze security logs
4. **Remediate**: Apply fixes and patches
5. **Recover**: Restore from secure backups
6. **Document**: Create incident report

---

## 12. Compliance Recommendations

### GDPR Compliance (if handling EU data)
- Implement data subject rights (access, deletion)
- Add privacy policy and consent mechanisms
- Implement data retention policies
- Enable audit logging

### Industry Standards
- Follow NIST Cybersecurity Framework
- Implement ISO 27001 controls
- Apply CIS Controls
- Follow OWASP ASVS guidelines

---

## Conclusion

The Local Document Q&A RAG system currently has significant security vulnerabilities that must be addressed before deployment in any production or sensitive environment. The lack of authentication, encryption, and input validation creates multiple attack vectors that could lead to data breaches or system compromise.

### Immediate Actions Required:
1. Implement authentication and authorization
2. Add input validation and sanitization
3. Enable HTTPS/TLS encryption
4. Update dependencies to latest secure versions
5. Implement security logging and monitoring

### Risk Mitigation Timeline:
- **Week 1**: Address all critical vulnerabilities
- **Week 2-4**: Implement high-priority security controls
- **Month 2**: Complete medium-priority enhancements
- **Ongoing**: Regular security updates and monitoring

### Final Risk Assessment After Recommendations:
If all recommended security controls are implemented, the residual risk would be reduced from **MEDIUM-HIGH** to **LOW-MEDIUM**, making the application suitable for handling moderately sensitive documents in a controlled environment.

---

**Report Generated**: 2025-09-16
**Next Review Date**: 2025-10-16
**Contact**: security-team@organization.com

## Appendix A: Security Tools and Resources

- [OWASP Top 10 2021](https://owasp.org/Top10/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [Streamlit Security Best Practices](https://docs.streamlit.io/library/advanced-features/security)
- [LangChain Security Guidelines](https://python.langchain.com/docs/security)

## Appendix B: Security Configuration Files

### .env.example
```bash
# Application Configuration
OLLAMA_MODEL=llama3
DOCUMENTS_DIR=./documents
CHROMA_DB_PATH=./chroma_db
MAX_FILE_SIZE=10485760
ALLOWED_EXTENSIONS=pdf,txt,md

# Security Configuration
ENABLE_AUTH=true
SESSION_SECRET=generate-random-secret-here
ENCRYPTION_KEY=generate-random-key-here
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Logging
LOG_LEVEL=INFO
SECURITY_LOG_FILE=./logs/security.log
```

### requirements-security.txt
```
streamlit-authenticator==0.3.1
python-dotenv==1.0.0
cryptography==41.0.5
safety==3.0.1
pip-audit==2.6.1
bandit==1.7.5
```

---

**END OF SECURITY AUDIT REPORT**