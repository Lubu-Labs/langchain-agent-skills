# LLM Provider Configuration

Guide to configuring API keys and credentials for various LLM providers.

## Quick Setup

Use the interactive setup script:

```bash
uv run scripts/setup_providers.py

# Fallback if uv not available
python3 scripts/setup_providers.py
```

Or manually edit `.env` file.

## OpenAI

### API Key
Get your key: https://platform.openai.com/api-keys

### Configuration
```bash
OPENAI_API_KEY=sk-...
```

### Usage
```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini")
```

## Anthropic (Claude)

### API Key
Get your key: https://console.anthropic.com/settings/keys

### Configuration
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### Usage
```python
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-haiku-4-5-20251001")
```

## Google (Gemini)

### API Key
Get your key: https://aistudio.google.com/app/apikey

### Configuration
```bash
GOOGLE_API_KEY=...
```

### Usage
```python
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-3.0-flash")
```

## AWS Bedrock

### Credentials
Configure AWS credentials: https://docs.aws.amazon.com/bedrock/

### Configuration
```bash
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

### Usage
```python
from langchain_aws import ChatBedrock
model = ChatBedrock(model_id="anthropic.claude-haiku-4-5-20251001-v1:0")
```

## LangSmith (Tracing)

### API Key
Get your key: https://smith.langchain.com/settings

### Configuration
```bash
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=my-project
```

### Optional Settings
```bash
# Use LANGSMITH_ENDPOINT only for self-hosted or non-default instances.
# Example: https://your-langsmith-host/api/v1
LANGSMITH_ENDPOINT=...
LANGSMITH_WORKSPACE_ID=...  # For multi-workspace keys
```

## Security Best Practices

1. **Never commit .env files** - Add to `.gitignore`
2. **Use environment-specific files** - `.env.local`, `.env.production`
3. **Rotate keys regularly** - Especially after sharing code
4. **Use secrets managers** - For production deployments
5. **Limit key permissions** - Use least-privilege principle

## Testing Configuration

Verify your setup:

```bash
langgraph dev
```

Check the logs for successful provider initialization.
