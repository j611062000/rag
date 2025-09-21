# 🔐 Security & API Key Management

## ✅ **Safe to Commit `.env` - Uses OS Environment Variables!**

### **🔒 New Security Model:**
✅ `.env` contains only placeholder values (safe to commit)
✅ Real secrets come from OS environment variables
✅ No accidental secret commits possible
✅ Works in all environments (dev/staging/prod)

### **🔑 Setting Up Your API Keys Securely**

#### **Method 1: OS Environment Variables (Recommended)**
```bash
# Set in your current shell session
export ANTHROPIC_API_KEY="sk-ant-api03-your-actual-key-here"
export OPENAI_API_KEY="sk-your-openai-key-here"
export TAVILY_API_KEY="tvly-your-tavily-key-here"

# Make permanent by adding to your shell profile
echo 'export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"' >> ~/.bashrc
echo 'export OPENAI_API_KEY="sk-your-openai-key-here"' >> ~/.bashrc

# Reload your shell
source ~/.bashrc
```

#### **Method 2: Temporary Session (Development)**
```bash
# Set for current session only
ANTHROPIC_API_KEY="your-key-here" ./run_tests.sh

# Or export before running
export ANTHROPIC_API_KEY="your-key-here"
docker-compose up
```

#### **Method 3: Docker Environment (Production)**
```bash
# In docker-compose.yml or docker run:
environment:
  - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
  - OPENAI_API_KEY=${OPENAI_API_KEY}

# Or use Docker secrets for production
docker secret create anthropic_key -
docker service create --secret anthropic_key ...
```

### **🔍 How to Check Your Setup**

```bash
# Check if environment variables are set
echo "Anthropic key set: $([ -n "$ANTHROPIC_API_KEY" ] && echo "✅ Yes" || echo "❌ No")"
echo "OpenAI key set: $([ -n "$OPENAI_API_KEY" ] && echo "✅ Yes" || echo "❌ No")"

# Test if service can access keys
curl http://localhost:8000/health

# Test with a real question (requires real API key)
curl -X POST http://localhost:8000/ask -d '{"question": "Hello"}' -H "Content-Type: application/json"
```

### **🌍 Different Environments**

```bash
# Development
.env                # Your local development keys

# Staging
.env.staging        # Staging environment keys

# Production
.env.production     # Production keys (use proper secret management)
```

### **⚠️  Security Best Practices**

1. **Never commit** `.env` files to version control
2. **Use different keys** for development/staging/production
3. **Rotate keys regularly** (every 90 days)
4. **Restrict key permissions** in Anthropic Console
5. **Monitor key usage** for unusual activity
6. **Use secrets management** in production (AWS Secrets Manager, etc.)

### **🚨 If You Accidentally Commit Keys**

1. **Immediately revoke** the key in Anthropic Console
2. **Generate a new key**
3. **Remove from git history**:
   ```bash
   git filter-branch --force --index-filter \
   'git rm --cached --ignore-unmatch .env' \
   --prune-empty --tag-name-filter cat -- --all
   ```

### **🔧 Testing Without Real Keys**

The system works with **mock providers** by default:
- No keys needed for development/testing
- Set `DEBUG=true` to use mocks
- Perfect for CI/CD pipelines

```bash
# Run without any API keys
DEBUG=true docker-compose up
```

### **🚀 Quick Start with Real API Key**

The recommended way to run the system:

```bash
# Method 1: Export then run
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
./run_with_api_key.sh

# Method 2: One-line execution
ANTHROPIC_API_KEY="sk-ant-api03-your-key-here" ./run_with_api_key.sh

# Method 3: Manual docker-compose (for advanced users)
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
cd docker && docker-compose up -d
```

**✅ Secure:** No API keys are ever stored in files or committed to version control.