#!/bin/bash
# Launch the Telegram AI Dashboard
# Prerequisites: telegram_api.py must be running on port 8765

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Telegram AI — Dashboard Launcher"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if telegram_api.py is reachable
if command -v curl &> /dev/null; then
    if curl -s --connect-timeout 2 http://localhost:8765/health > /dev/null 2>&1; then
        echo "  ✓ Bot API is running on localhost:8765"
    else
        echo "  ⚠ Bot API not detected on localhost:8765"
        echo "    Start it first: uv run python telegram_api.py"
        echo ""
    fi
fi

echo "  Starting dashboard..."
echo ""

# Auto-open browser after a short delay
(sleep 4 && open "http://localhost:3000" 2>/dev/null || xdg-open "http://localhost:3000" 2>/dev/null) &

# Run the Reflex app
uv run reflex run
