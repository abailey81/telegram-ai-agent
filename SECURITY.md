# Security Policy

## Supported Versions

| Version | Supported |
|:--------|:---------:|
| Latest `main` | Yes |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public issue
2. Email the maintainer directly or use [GitHub's private vulnerability reporting](https://github.com/abailey81/telegram-ai-agent/security/advisories/new)
3. Include a description of the vulnerability, steps to reproduce, and potential impact

You can expect an initial response within 72 hours.

## Security Considerations

This project handles Telegram API credentials and user conversation data. Key security measures:

- **Credentials**: All secrets stored in `.env` (never committed — listed in `.gitignore`)
- **Session strings**: Telegram session strings provide full account access — treat as passwords
- **User data**: Conversation logs, memory, and RL data stored locally in `engine_data/` and `rl_data/` (excluded from version control)
- **Docker**: Runs as non-root user (`appuser`) inside the container
- **Input validation**: All chat/user ID parameters validated via `@validate_id()` decorator

## Best Practices

- Never commit `.env` files or session strings
- Rotate Telegram session strings if compromised
- Use Docker for production deployments (isolated environment)
- Review `engine_data/` contents periodically — it contains per-user interaction data
