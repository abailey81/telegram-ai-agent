/**
 * Configuration for the Telegram AI Agent
 * Loads environment from parent directory's .env file
 */

import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

// Load .env from parent directory (telegram-mcp root)
const __dirname = dirname(fileURLToPath(import.meta.url));
const envPath = resolve(__dirname, "../..", ".env");

try {
  const content = await Bun.file(envPath).text();
  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const [key, ...valueParts] = trimmed.split("=");
    if (key && valueParts.length > 0) {
      const value = valueParts.join("=").trim();
      if (!process.env[key]) {
        process.env[key] = value;
      }
    }
  }
} catch (e) {
  console.error(`Warning: Could not load .env from ${envPath}`);
}

export const config = {
  // Telegram HTTP Bridge
  telegramApiUrl: process.env.TELEGRAM_API_URL || "http://localhost:8765",

  // Nia API
  niaApiKey: process.env.NIA_API_KEY || "",
  niaApiBase: "https://apigcp.trynia.ai/v2",
  niaCodebaseSource: process.env.NIA_CODEBASE_SOURCE || "",

  // AI Model - configurable via AGENT_MODEL env var, defaults to Haiku 4.5 (fast + cheap)
  model: process.env.AGENT_MODEL || "claude-haiku-4-5-20251001",
} as const;

export function validateConfig() {
  const missing: string[] = [];

  if (!process.env.ANTHROPIC_API_KEY) {
    missing.push("ANTHROPIC_API_KEY");
  }
  if (!config.niaApiKey) {
    missing.push("NIA_API_KEY");
  }

  if (missing.length > 0) {
    console.error(`❌ Missing required environment variables: ${missing.join(", ")}`);
    console.error("   Make sure your .env file in the project root has the keys");
    process.exit(1);
  }
}
