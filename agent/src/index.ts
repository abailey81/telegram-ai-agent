#!/usr/bin/env bun
/**
 * Telegram AI Agent CLI
 * A stunning interactive terminal for communicating via Telegram with AI assistance
 */

import * as p from "@clack/prompts";
import pc from "picocolors";
import { chat, clearHistory, getHistoryLength } from "./agent";
import { config, validateConfig } from "./config";

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// VISUAL CONSTANTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const GRADIENT_CHARS = "░▒▓█";

function gradient(text: string, colors: ((s: string) => string)[]): string {
  const chars = text.split("");
  return chars
    .map((char, i) => {
      const colorIdx = Math.floor((i / chars.length) * colors.length);
      return colors[Math.min(colorIdx, colors.length - 1)](char);
    })
    .join("");
}

// Beautiful ASCII art banner with gradient
const BANNER = `
${pc.dim("  ┌─────────────────────────────────────────────────────────────┐")}
${pc.dim("  │")}                                                             ${pc.dim("│")}
${pc.dim("  │")}   ${pc.bold(pc.magenta("████████"))}${pc.bold(pc.cyan("╗"))} ${pc.bold(pc.magenta("███████"))}${pc.bold(pc.cyan("╗"))} ${pc.bold(pc.magenta("██"))}${pc.bold(pc.cyan("╗"))}     ${pc.bold(pc.magenta("██"))}${pc.bold(pc.cyan("╗"))}  ${pc.bold(pc.magenta("██"))}${pc.bold(pc.cyan("╗"))} ${pc.bold(pc.magenta("███████"))}${pc.bold(pc.cyan("╗"))}     ${pc.dim("│")}
${pc.dim("  │")}   ${pc.dim("╚══")}${pc.magenta("██")}${pc.dim("╔══╝")} ${pc.magenta("██")}${pc.dim("╔════╝")} ${pc.magenta("██")}${pc.dim("║")}     ${pc.magenta("██")}${pc.dim("║")}  ${pc.magenta("██")}${pc.dim("║")} ${pc.magenta("██")}${pc.dim("╔════╝")}     ${pc.dim("│")}
${pc.dim("  │")}      ${pc.cyan("██")}${pc.dim("║")}    ${pc.cyan("█████")}${pc.dim("╗")}   ${pc.cyan("██")}${pc.dim("║")}     ${pc.cyan("██")}${pc.dim("║")}  ${pc.cyan("██")}${pc.dim("║")} ${pc.cyan("█████")}${pc.dim("╗")}       ${pc.dim("│")}
${pc.dim("  │")}      ${pc.blue("██")}${pc.dim("║")}    ${pc.blue("██")}${pc.dim("╔══╝")}   ${pc.blue("██")}${pc.dim("║")}     ${pc.blue("██")}${pc.dim("║")}  ${pc.blue("██")}${pc.dim("║")} ${pc.blue("██")}${pc.dim("╔══╝")}       ${pc.dim("│")}
${pc.dim("  │")}      ${pc.blue("██")}${pc.dim("║")}    ${pc.blue("███████")}${pc.dim("╗")} ${pc.blue("███████")}${pc.dim("╗")} ${pc.dim("╚")}${pc.blue("████")}${pc.dim("╔╝")} ${pc.blue("███████")}${pc.dim("╗")}     ${pc.dim("│")}
${pc.dim("  │")}      ${pc.dim("╚═╝    ╚══════╝ ╚══════╝  ╚═══╝  ╚══════╝")}     ${pc.dim("│")}
${pc.dim("  │")}                                                             ${pc.dim("│")}
${pc.dim("  │")}    ${pc.bold(pc.white("Telegram AI Agent"))} ${pc.dim("·")} ${pc.italic(pc.cyan("Conversational Intelligence Platform"))}            ${pc.dim("│")}
${pc.dim("  │")}                                                             ${pc.dim("│")}
${pc.dim("  └─────────────────────────────────────────────────────────────┘")}
`;

const DIVIDER = pc.dim("  ─────────────────────────────────────────────────────────────");
const THIN_DIVIDER = pc.dim("  · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·");

// Help text — styled
const HELP_TEXT = `
${DIVIDER}
${pc.bold(pc.white("  CORE COMMANDS"))}
${DIVIDER}

  ${pc.bgMagenta(pc.white(" /help "))}       ${pc.dim("─")} Show this help menu
  ${pc.bgCyan(pc.white(" /clear "))}      ${pc.dim("─")} Clear conversation history
  ${pc.bgBlue(pc.white(" /status "))}     ${pc.dim("─")} Full system status
  ${pc.bgYellow(pc.black(" /chats "))}      ${pc.dim("─")} Recent chats with previews

${THIN_DIVIDER}
${pc.bold(pc.white("  INTERVENTION & CONTROL"))}

  ${pc.bgRed(pc.white(" /pause "))}      ${pc.dim("─")} Pause auto-reply       ${pc.dim("/pause @user [min]")}
  ${pc.bgGreen(pc.white(" /resume "))}     ${pc.dim("─")} Resume auto-reply      ${pc.dim("/resume @user")}
  ${pc.bgMagenta(pc.white(" /tell "))}       ${pc.dim("─")} Tell bot what to do     ${pc.dim("/tell @user be concise")}
  ${pc.bgCyan(pc.white(" /send "))}       ${pc.dim("─")} Send message directly  ${pc.dim("/send @user hey there")}
  ${pc.bgBlue(pc.white(" /queue "))}      ${pc.dim("─")} Queue delayed message  ${pc.dim("/queue @user 60 checking in")}
  ${pc.bgYellow(pc.black(" /intervene "))} ${pc.dim("─")} Show active overrides

${THIN_DIVIDER}
${pc.bold(pc.white("  VOICE CLONING"))}

  ${pc.bgMagenta(pc.white(" /voice "))}      ${pc.dim("─")} Voice engine status
  ${pc.bgCyan(pc.white(" /voice register "))} ${pc.dim("─")} Register voice     ${pc.dim("/voice register saved")}
  ${pc.bgBlue(pc.white(" /voice send "))}    ${pc.dim("─")} Send voice msg     ${pc.dim("/voice send @user hey")}
  ${pc.bgYellow(pc.black(" /voice list "))}    ${pc.dim("─")} List all voices
  ${pc.bgMagenta(pc.white(" /voice assign "))}  ${pc.dim("─")} Per-chat voice     ${pc.dim("/voice assign @user path")}

${THIN_DIVIDER}
${pc.bold(pc.white("  VOICE CALLS"))}

  ${pc.bgRed(pc.white(" /call "))}        ${pc.dim("─")} Call status / make call  ${pc.dim("/call @user")}
  ${pc.bgGreen(pc.white(" /call answer "))} ${pc.dim("─")} Accept incoming call    ${pc.dim("/call answer @user")}
  ${pc.bgYellow(pc.black(" /call hangup "))} ${pc.dim("─")} End active call         ${pc.dim("/call hangup @user")}
  ${pc.bgCyan(pc.white(" /call speak "))}  ${pc.dim("─")} Speak in call           ${pc.dim("/call speak @user <text>")}
  ${pc.bgBlue(pc.white(" /call listen "))} ${pc.dim("─")} Transcribe call audio   ${pc.dim("/call listen @user")}
  ${pc.bgMagenta(pc.white(" /call auto "))}   ${pc.dim("─")} Toggle full autonomy    ${pc.dim("/call auto @user")}
  ${pc.bgCyan(pc.white(" /call auto off "))} ${pc.dim("─")} Disable autonomy       ${pc.dim("/call auto off @user")}
  ${pc.bgYellow(pc.black(" /call autoanswer "))} ${pc.dim("─")} Auto-accept calls   ${pc.dim("/call autoanswer on")}
  ${pc.bgBlue(pc.white(" /call group "))}  ${pc.dim("─")} Group voice chat        ${pc.dim("/call group join @chat")}
  ${pc.bgRed(pc.white(" /call bridge "))} ${pc.dim("─")} Start call bridge

${THIN_DIVIDER}
${pc.bold(pc.white("  SYSTEM"))}

  ${pc.bgMagenta(pc.white(" /dashboard "))} ${pc.dim("─")} Live system dashboard
  ${pc.bgCyan(pc.white(" /features "))}  ${pc.dim("─")} Feature flags ${pc.dim("(toggle: /features toggle <name>)")}
  ${pc.bgBlue(pc.white(" /engines "))}   ${pc.dim("─")} Intelligence engine status
  ${pc.bgYellow(pc.black(" /models "))}    ${pc.dim("─")} ML model accuracies
  ${pc.bgMagenta(pc.white(" /log "))}       ${pc.dim("─")} Recent auto-reply activity

${THIN_DIVIDER}
${pc.bold(pc.white("  INTELLIGENCE"))}

  ${pc.bgCyan(pc.white(" /analyze "))}   ${pc.dim("─")} V5 psychological analysis  ${pc.dim("/analyze @user")}
  ${pc.bgBlue(pc.white(" /health "))}    ${pc.dim("─")} Conversation health score  ${pc.dim("/health @user")}
  ${pc.bgYellow(pc.black(" /memory "))}    ${pc.dim("─")} Memory recall              ${pc.dim("/memory @user")}
  ${pc.bgMagenta(pc.white(" /train "))}     ${pc.dim("─")} Train ML models

${THIN_DIVIDER}

  ${pc.bgRed(pc.white(" /quit "))}      ${pc.dim("─")} Exit the agent

${DIVIDER}
${pc.bold(pc.white("  EXAMPLES"))}
${DIVIDER}

  ${pc.cyan(">")} ${pc.white("Read the last 5 messages from @username")}
  ${pc.cyan(">")} ${pc.white("What should I reply to their message?")}
  ${pc.cyan(">")} ${pc.white("/tell @user be more concise for 30 minutes")}
  ${pc.cyan(">")} ${pc.white("/pause @user 60")} ${pc.dim("(take manual control for 1 hour)")}
  ${pc.cyan(">")} ${pc.white("/send @user hey, how's it going?")}
  ${pc.cyan(">")} ${pc.white("/voice register saved")} ${pc.dim("(clone from Saved Messages)")}
  ${pc.cyan(">")} ${pc.white("/voice send @user thanks for the update")}
  ${pc.cyan(">")} ${pc.white("/queue @user 300 hey, just checking in")}
  ${pc.cyan(">")} ${pc.white("Be friendly with @username")} ${pc.dim("(sets auto-reply tone)")}

${DIVIDER}
`;

// Status indicators
const STATUS = {
  online: `${pc.green("●")} ${pc.green("Online")}`,
  offline: `${pc.red("●")} ${pc.red("Offline")}`,
  connecting: `${pc.yellow("◐")} ${pc.yellow("Connecting...")}`,
};

// Thinking animation frames
const THINKING_FRAMES = ["◐", "◓", "◑", "◒"];

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// API HELPERS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async function apiCall<T = any>(
  endpoint: string,
  method = "GET",
  body?: any
): Promise<T> {
  const response = await fetch(`${config.telegramApiUrl}${endpoint}`, {
    method,
    headers: { "Content-Type": "application/json" },
    ...(body !== undefined && { body: JSON.stringify(body) }),
  });
  if (!response.ok) throw new Error(`API ${response.status}`);
  return response.json();
}

async function checkTelegramConnection(): Promise<boolean> {
  try {
    const response = await fetch(`${config.telegramApiUrl}/health`);
    if (response.ok) {
      const data = (await response.json()) as { connected?: boolean };
      return data.connected === true;
    }
    return false;
  } catch {
    return false;
  }
}

async function fetchAutoReplyStatus(): Promise<string> {
  try {
    const response = await fetch(`${config.telegramApiUrl}/auto-reply/status`);
    if (response.ok) {
      const data = (await response.json()) as {
        enabled?: boolean;
        chat_ids?: string[];
        features?: Record<string, boolean>;
      };
      if (data.enabled) {
        const chats = data.chat_ids?.length || 0;
        return `${pc.green("ON")} ${pc.dim(`(${chats} chats)`)}`;
      }
      return pc.dim("OFF");
    }
    return pc.dim("unknown");
  } catch {
    return pc.dim("unavailable");
  }
}

async function fetchEngineStatus(): Promise<string> {
  try {
    const response = await fetch(`${config.telegramApiUrl}/engine/status`);
    if (response.ok) {
      const data = (await response.json()) as { engines?: Record<string, unknown> };
      const count = Object.keys(data.engines || {}).length;
      return `${pc.green(`${count}`)} ${pc.dim("loaded")}`;
    }
    return pc.dim("—");
  } catch {
    return pc.dim("—");
  }
}

function formatTimestamp(): string {
  const now = new Date();
  const h = now.getHours().toString().padStart(2, "0");
  const m = now.getMinutes().toString().padStart(2, "0");
  const s = now.getSeconds().toString().padStart(2, "0");
  return pc.dim(`${h}:${m}:${s}`);
}

// Color an accuracy value
function colorAccuracy(acc: number): string {
  const pct = (acc * 100).toFixed(1);
  if (acc >= 0.93) return pc.green(`${pct}%`);
  if (acc >= 0.85) return pc.yellow(`${pct}%`);
  return pc.red(`${pct}%`);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// COMMAND RENDERERS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async function showDetailedStatus() {
  const connected = await checkTelegramConnection();
  const autoReply = await fetchAutoReplyStatus();
  const engines = await fetchEngineStatus();

  let modelSummary = pc.dim("—");
  let featureSummary = pc.dim("—");
  let dlDevice = pc.dim("—");

  if (connected) {
    try {
      const models = await apiCall<any>("/models/status");
      const sk = models.sklearn || [];
      const nn = models.neural || [];
      const avgAcc = sk.length > 0
        ? sk.reduce((s: number, m: any) => s + (m.accuracy || 0), 0) / sk.length
        : 0;
      modelSummary = `${pc.green(`${sk.length}`)} sklearn ${pc.dim("+")} ${pc.green(`${nn.length}`)} neural ${avgAcc > 0 ? pc.dim(`(avg ${(avgAcc * 100).toFixed(0)}%)`) : ""}`;
    } catch {}

    try {
      const features = await apiCall<any>("/auto-reply/features");
      const keys = Object.keys(features).filter(k => k !== "current_hour" && k !== "late_night_active" && k !== "proactive_max_per_day");
      const active = keys.filter(k => features[k] === true).length;
      featureSummary = `${pc.green(`${active}`)}${pc.dim(`/${keys.length}`)} ${pc.dim("active")}`;
    } catch {}

    try {
      const dl = await apiCall<any>("/dl/status");
      dlDevice = pc.cyan(dl.device || "cpu");
    } catch {}
  }

  console.log("");
  console.log(DIVIDER);
  console.log(pc.bold(pc.white("  SYSTEM STATUS")));
  console.log(DIVIDER);
  console.log("");
  console.log(`  ${pc.dim("Telegram")}      ${connected ? STATUS.online : STATUS.offline}`);
  console.log(`  ${pc.dim("API Bridge")}    ${connected ? pc.green("localhost:8765") : pc.red("not running")}`);
  console.log(`  ${pc.dim("Auto-Reply")}    ${autoReply}`);
  console.log(`  ${pc.dim("Features")}      ${featureSummary}`);
  console.log(`  ${pc.dim("AI Engines")}    ${engines}`);
  console.log(`  ${pc.dim("ML Models")}     ${modelSummary}`);
  console.log(`  ${pc.dim("DL Device")}     ${dlDevice}`);
  console.log(`  ${pc.dim("Model")}         ${pc.cyan(config.model)}`);
  console.log(`  ${pc.dim("History")}       ${pc.white(String(getHistoryLength()))} ${pc.dim("messages")}`);
  console.log(`  ${pc.dim("Nia Search")}    ${config.niaCodebaseSource ? pc.green("configured") : pc.yellow("not set")}`);
  console.log("");
  console.log(DIVIDER);
  console.log("");
}

async function showDashboard() {
  try {
    const data = await apiCall<any>("/dashboard");
    console.log("");
    console.log(DIVIDER);
    console.log(pc.bold(pc.white("  SYSTEM DASHBOARD")));
    console.log(DIVIDER);

    // Auto-Reply
    const ar = data.auto_reply || {};
    console.log("");
    console.log(`  ${pc.bold(pc.magenta("AUTO-REPLY"))}   ${ar.enabled ? `${pc.green("●")} ${pc.green("ON")}` : `${pc.red("●")} ${pc.red("OFF")}`}  ${pc.dim(`${ar.chat_count || 0} chats`)}  ${pc.dim(`${ar.feature_count || 0} features active`)}`);

    // Engines
    const engines = data.engines || {};
    const engineNames = Object.keys(engines);
    if (engineNames.length > 0) {
      console.log("");
      console.log(`  ${pc.bold(pc.cyan("ENGINES"))}`);
      engineNames.forEach((name, i) => {
        const e = engines[name];
        const prefix = i < engineNames.length - 1 ? "├" : "└";
        const funcs = e.functions || 0;
        const status = e.status === "loaded" ? pc.green("●") : pc.red("●");
        console.log(`  ${pc.dim(prefix)} ${status} ${pc.white(name.padEnd(22))} ${pc.dim(`${funcs} functions`)}`);
      });
    }

    // Models
    const models = data.models || {};
    const sklearn = models.sklearn || [];
    const neural = models.neural || [];
    if (sklearn.length > 0 || neural.length > 0) {
      console.log("");
      console.log(`  ${pc.bold(pc.blue("MODELS"))}`);
      const allModels = [...sklearn, ...neural];
      allModels.forEach((m: any, i: number) => {
        const prefix = i < allModels.length - 1 ? "├" : "└";
        const acc = m.accuracy ? colorAccuracy(m.accuracy) : pc.dim("—");
        const type = (m.classifier_type || m.type || "").padEnd(5);
        console.log(`  ${pc.dim(prefix)} ${pc.white(m.name.padEnd(22))} ${pc.dim(type)} ${acc}`);
      });
    }

    // Recent Activity
    const recent = data.recent_activity || [];
    if (recent.length > 0) {
      console.log("");
      console.log(`  ${pc.bold(pc.yellow("RECENT ACTIVITY"))}`);
      recent.forEach((entry: any, i: number) => {
        const prefix = i < recent.length - 1 ? "├" : "└";
        const time = entry.time || "";
        const msg = entry.message || "";
        console.log(`  ${pc.dim(prefix)} ${pc.dim(time)} ${msg.slice(0, 60)}`);
      });
    }

    console.log("");
    console.log(DIVIDER);
    console.log("");
  } catch (e: any) {
    console.log(`  ${pc.red("!")} ${pc.dim("Dashboard unavailable — is the backend running?")}`);
    console.log("");
  }
}

async function showFeatures() {
  try {
    const data = await apiCall<any>("/auto-reply/features");
    console.log("");
    console.log(DIVIDER);
    console.log(pc.bold(pc.white("  FEATURE FLAGS")));
    console.log(DIVIDER);
    console.log("");

    const keys = Object.keys(data).filter(k => k !== "current_hour" && k !== "late_night_active" && k !== "proactive_max_per_day");

    // Render in 2 columns
    for (let i = 0; i < keys.length; i += 2) {
      const k1 = keys[i];
      const v1 = data[k1];
      const badge1 = v1 ? `${pc.green("●")} ${pc.white(k1.padEnd(24))}` : `${pc.dim("○")} ${pc.dim(k1.padEnd(24))}`;

      let col2 = "";
      if (i + 1 < keys.length) {
        const k2 = keys[i + 1];
        const v2 = data[k2];
        col2 = v2 ? `${pc.green("●")} ${pc.white(k2)}` : `${pc.dim("○")} ${pc.dim(k2)}`;
      }

      console.log(`  ${badge1} ${col2}`);
    }

    // Info row
    if (data.proactive_max_per_day !== undefined) {
      console.log("");
      console.log(`  ${pc.dim("proactive_max_per_day =")} ${pc.cyan(String(data.proactive_max_per_day))}`);
    }
    if (data.late_night_active !== undefined) {
      console.log(`  ${pc.dim("late_night_active =")} ${data.late_night_active ? pc.yellow("yes") : pc.dim("no")} ${pc.dim(`(hour: ${data.current_hour})`)}`);
    }

    console.log("");
    console.log(`  ${pc.dim("Toggle:")} ${pc.cyan("/features toggle <name>")}`);
    console.log("");
    console.log(DIVIDER);
    console.log("");
  } catch {
    console.log(`  ${pc.red("!")} ${pc.dim("Features unavailable")}`);
    console.log("");
  }
}

async function handleFeatureToggle(featureName: string) {
  try {
    // Get current value
    const features = await apiCall<any>("/auto-reply/features");
    if (!(featureName in features)) {
      console.log(`  ${pc.red("!")} ${pc.dim("Unknown feature:")} ${pc.white(featureName)}`);
      console.log(`  ${pc.dim("Available:")} ${Object.keys(features).filter(k => k !== "current_hour" && k !== "late_night_active" && k !== "proactive_max_per_day").join(", ")}`);
      console.log("");
      return;
    }

    const newValue = !features[featureName];
    await apiCall("/auto-reply/features", "PUT", { [featureName]: newValue });

    const icon = newValue ? pc.green("●") : pc.dim("○");
    const state = newValue ? pc.green("ON") : pc.dim("OFF");
    console.log(`  ${icon} ${pc.white(featureName)} ${pc.dim("→")} ${state}`);
    console.log("");
  } catch {
    console.log(`  ${pc.red("!")} ${pc.dim("Failed to toggle feature")}`);
    console.log("");
  }
}

async function showEngines() {
  try {
    const data = await apiCall<any>("/engine/status");
    const engines = data.engines || {};

    console.log("");
    console.log(DIVIDER);
    console.log(pc.bold(pc.white("  INTELLIGENCE ENGINES")));
    console.log(DIVIDER);
    console.log("");

    const names = Object.keys(engines);
    if (names.length === 0) {
      console.log(`  ${pc.dim("No engines loaded")}`);
    } else {
      names.forEach((name, i) => {
        const e = engines[name];
        const status = e.status === "loaded" ? pc.green("●") : pc.red("●");
        const funcs = e.functions || [];
        const funcList = Array.isArray(funcs)
          ? funcs.map((f: string) => pc.dim(f)).join(pc.dim(", "))
          : pc.dim(`${funcs} functions`);

        console.log(`  ${status} ${pc.bold(pc.white(name))}`);
        console.log(`    ${funcList}`);
        if (i < names.length - 1) console.log("");
      });
    }

    console.log("");
    console.log(DIVIDER);
    console.log("");
  } catch {
    console.log(`  ${pc.red("!")} ${pc.dim("Engines unavailable")}`);
    console.log("");
  }
}

async function showLog() {
  try {
    const data = await apiCall<any>("/auto-reply/log?limit=15");
    const log = data.log || [];

    console.log("");
    console.log(DIVIDER);
    console.log(pc.bold(pc.white("  AUTO-REPLY LOG")));
    console.log(DIVIDER);
    console.log("");

    if (log.length === 0) {
      console.log(`  ${pc.dim("No recent activity")}`);
    } else {
      log.forEach((entry: any) => {
        const time = entry.timestamp ? pc.dim(new Date(entry.timestamp).toLocaleTimeString().slice(0, 5)) : pc.dim("--:--");
        const chat = entry.chat_id ? pc.cyan(String(entry.chat_id).padEnd(16)) : pc.dim("unknown".padEnd(16));

        if (entry.incoming) {
          console.log(`  ${time}  ${chat} ${pc.dim("←")} ${pc.white(`"${String(entry.incoming).slice(0, 45)}"`)}`)
        }
        if (entry.reply) {
          console.log(`  ${pc.dim("".padEnd(7))} ${pc.dim("".padEnd(16))} ${pc.green("→")} ${pc.green(`"${String(entry.reply).slice(0, 45)}"`)}`);
        }
        if (entry.reaction) {
          console.log(`  ${time}  ${chat} ${pc.yellow("♥")} ${entry.reaction}`);
        }
        if (!entry.incoming && !entry.reply && !entry.reaction) {
          // Generic log entry
          const msg = entry.message || JSON.stringify(entry);
          console.log(`  ${time}  ${msg.slice(0, 60)}`);
        }
      });
    }

    console.log("");
    console.log(DIVIDER);
    console.log("");
  } catch {
    console.log(`  ${pc.red("!")} ${pc.dim("Log unavailable")}`);
    console.log("");
  }
}

async function showModels() {
  try {
    const data = await apiCall<any>("/models/status");

    console.log("");
    console.log(DIVIDER);
    console.log(pc.bold(pc.white("  ML MODELS")));
    console.log(DIVIDER);

    // Sklearn
    const sklearn = data.sklearn || [];
    if (sklearn.length > 0) {
      console.log("");
      console.log(`  ${pc.bold(pc.cyan("SKLEARN CLASSIFIERS"))}`);
      sklearn.forEach((m: any, i: number) => {
        const prefix = i < sklearn.length - 1 ? "├" : "└";
        const name = (m.name || "unknown").padEnd(22);
        const type = (m.classifier_type || "—").padEnd(6);
        const acc = m.accuracy ? colorAccuracy(m.accuracy) : pc.dim("—");
        const classes = m.class_count ? pc.dim(`${m.class_count} classes`) : "";
        const samples = m.training_size ? pc.dim(`${m.training_size} samples`) : "";
        console.log(`  ${pc.dim(prefix)} ${pc.white(name)} ${pc.dim(type)} ${acc}  ${classes}  ${samples}`);
      });
    }

    // Neural
    const neural = data.neural || [];
    if (neural.length > 0) {
      console.log("");
      console.log(`  ${pc.bold(pc.magenta("NEURAL NETWORKS"))}`);
      neural.forEach((m: any, i: number) => {
        const prefix = i < neural.length - 1 ? "├" : "└";
        const name = (m.name || "unknown").padEnd(22);
        const type = (m.type || "—").padEnd(6);
        const acc = m.accuracy ? colorAccuracy(m.accuracy) : pc.dim("—");
        const classes = m.class_count ? pc.dim(`${m.class_count} classes`) : "";
        console.log(`  ${pc.dim(prefix)} ${pc.white(name)} ${pc.dim(type)} ${acc}  ${classes}`);
      });
    }

    if (sklearn.length === 0 && neural.length === 0) {
      console.log("");
      console.log(`  ${pc.dim("No trained models found")}`);
    }

    console.log("");
    console.log(DIVIDER);
    console.log("");
  } catch {
    console.log(`  ${pc.red("!")} ${pc.dim("Models unavailable")}`);
    console.log("");
  }
}

async function showChats() {
  try {
    const data = await apiCall<any>("/chats?limit=20");
    const chats = data.chats || [];

    console.log("");
    console.log(DIVIDER);
    console.log(pc.bold(pc.white("  RECENT CHATS")));
    console.log(DIVIDER);
    console.log("");

    if (chats.length === 0) {
      console.log(`  ${pc.dim("No chats found")}`);
    } else {
      chats.forEach((c: any) => {
        const icon = c.type === "channel" ? "📢" : c.type === "chat" ? "👥" : "👤";
        const name = c.title || `${c.first_name || ""} ${c.last_name || ""}`.trim() || "Unknown";
        const username = c.username ? pc.dim(` @${c.username}`) : "";
        const unread = c.unread_count ? `  ${pc.bgRed(pc.white(` ${c.unread_count} `))}` : "";
        const preview = c.last_message ? `\n      ${pc.dim(`"${c.last_message.slice(0, 55)}"${c.last_message.length > 55 ? "..." : ""}`)}` : "";

        console.log(`  ${icon}  ${pc.bold(pc.white(name))}${username}${unread}${preview}`);
        console.log("");
      });
    }

    console.log(DIVIDER);
    console.log("");
  } catch {
    console.log(`  ${pc.red("!")} ${pc.dim("Chats unavailable — is the backend running?")}`);
    console.log("");
  }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MAIN
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async function main() {
  console.clear();
  console.log(BANNER);

  // Validate configuration
  validateConfig();

  // Startup animation
  const connectionSpinner = p.spinner();
  connectionSpinner.start(pc.dim("Connecting to Telegram..."));

  const isConnected = await checkTelegramConnection();

  if (isConnected) {
    connectionSpinner.stop(
      `  ${pc.green("●")} Telegram ${pc.green("connected")} ${pc.dim("·")} API on ${pc.cyan("localhost:8765")}`
    );
  } else {
    connectionSpinner.stop(`  ${pc.red("●")} Telegram ${pc.red("not connected")}`);
    console.log("");
    console.log(`  ${pc.yellow("!")} Start the API bridge first:`);
    console.log(`  ${pc.dim("$")} ${pc.cyan("uv run python telegram_api.py")}`);
    console.log("");
  }

  // Fetch additional status
  if (isConnected) {
    const autoReply = await fetchAutoReplyStatus();
    const engines = await fetchEngineStatus();

    let modelInfo = "";
    try {
      const models = await apiCall<any>("/models/status");
      const sk = models.sklearn?.length || 0;
      const nn = models.neural?.length || 0;
      modelInfo = `${sk + nn} trained`;
    } catch {
      modelInfo = pc.dim("—");
    }

    console.log("");
    console.log(`  ${pc.dim("├")} Auto-Reply  ${autoReply}`);
    console.log(`  ${pc.dim("├")} AI Engines  ${engines}`);
    console.log(`  ${pc.dim("├")} ML Models   ${pc.green(modelInfo)}`);
    console.log(`  ${pc.dim("└")} Model       ${pc.cyan(config.model)}`);
  }

  console.log("");
  console.log(THIN_DIVIDER);
  console.log(`  ${pc.dim("Type")} ${pc.yellow("/help")} ${pc.dim("for commands or just start chatting")}`);
  console.log(THIN_DIVIDER);
  console.log("");

  // Main chat loop
  while (true) {
    const input = await p.text({
      message: `${formatTimestamp()} ${pc.bold(pc.cyan("you"))}`,
      placeholder: "ask me anything...",
    });

    // Handle cancellation (Ctrl+C)
    if (p.isCancel(input)) {
      console.log("");
      console.log(THIN_DIVIDER);
      console.log(`  ${pc.dim("See you later")} ${pc.magenta("♥")}`);
      console.log(THIN_DIVIDER);
      console.log("");
      process.exit(0);
    }

    const message = (input as string).trim();
    if (!message) continue;

    // Handle commands
    let passToAgent = false;
    let agentOverride = "";

    if (message.startsWith("/")) {
      const parts = message.split(" ");
      const cmd = parts[0].toLowerCase();

      switch (cmd) {
        case "/help":
          console.log(HELP_TEXT);
          continue;

        case "/clear":
          clearHistory();
          console.log(`  ${pc.green("✓")} ${pc.dim("History cleared")}`);
          console.log("");
          continue;

        case "/status":
          await showDetailedStatus();
          continue;

        case "/dashboard":
          await showDashboard();
          continue;

        case "/features": {
          if (parts.length >= 3 && parts[1].toLowerCase() === "toggle") {
            await handleFeatureToggle(parts[2]);
          } else {
            await showFeatures();
          }
          continue;
        }

        case "/engines":
          await showEngines();
          continue;

        case "/models":
          await showModels();
          continue;

        case "/log":
          await showLog();
          continue;

        case "/chats":
          await showChats();
          continue;

        // ─── INTERVENTION & CONTROL ───────────────────────────
        case "/pause": {
          const args = parts.slice(1);
          if (!args[0]) {
            console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/pause @user [minutes]")}`);
            console.log("");
            continue;
          }
          const target = args[0];
          const mins = parseInt(args[1]) || 30;
          try {
            const params = new URLSearchParams({ minutes: String(mins) });
            const result = await apiCall<any>(`/intervene/${target}/pause?${params}`, "POST");
            console.log(`  ${pc.yellow("⏸")} ${pc.white(target)} paused for ${pc.cyan(String(mins))}min`);
            if (result.resumes_at) console.log(`  ${pc.dim("Resumes at")} ${pc.white(result.resumes_at)}`);
          } catch {
            console.log(`  ${pc.red("!")} ${pc.dim("Failed to pause")}`);
          }
          console.log("");
          continue;
        }

        case "/resume": {
          const target = parts[1];
          if (!target) {
            console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/resume @user")}`);
            console.log("");
            continue;
          }
          try {
            await apiCall<any>(`/intervene/${target}/resume`, "POST");
            console.log(`  ${pc.green("▶")} ${pc.white(target)} auto-reply ${pc.green("resumed")}`);
          } catch {
            console.log(`  ${pc.red("!")} ${pc.dim("Failed to resume")}`);
          }
          console.log("");
          continue;
        }

        case "/tell": {
          const args = parts.slice(1);
          if (args.length < 2) {
            console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/tell @user <instruction>")}`);
            console.log(`  ${pc.dim("Example:")} /tell @user be more aggressive for 30 minutes`);
            console.log("");
            continue;
          }
          const target = args[0];
          const instruction = args.slice(1).join(" ");
          // Check if instruction ends with "for N minutes"
          const durMatch = instruction.match(/\bfor\s+(\d+)\s*min/i);
          const dur = durMatch ? parseInt(durMatch[1]) : 0;
          const cleanInstruction = durMatch ? instruction.replace(durMatch[0], "").trim() : instruction;
          try {
            const params = new URLSearchParams({
              instruction: cleanInstruction,
              duration_minutes: String(dur),
            });
            await apiCall<any>(`/intervene/${target}?${params}`, "POST");
            if (dur > 0) {
              console.log(`  ${pc.magenta("🎯")} ${pc.white(target)} → ${pc.cyan(cleanInstruction)} ${pc.dim(`(${dur}min)`)}`);
            } else {
              console.log(`  ${pc.magenta("🎯")} ${pc.white(target)} → ${pc.cyan(cleanInstruction)} ${pc.dim("(next reply only)")}`);
            }
          } catch {
            console.log(`  ${pc.red("!")} ${pc.dim("Failed to set intervention")}`);
          }
          console.log("");
          continue;
        }

        case "/send": {
          const args = parts.slice(1);
          if (args.length < 2) {
            console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/send @user <message>")}`);
            console.log("");
            continue;
          }
          passToAgent = true;
          agentOverride = `Send this exact message to ${args[0]}: "${args.slice(1).join(" ")}"`;
          break;
        }

        case "/queue": {
          const args = parts.slice(1);
          if (args.length < 3) {
            console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/queue @user <delay_sec> <message>")}`);
            console.log(`  ${pc.dim("Example:")} /queue @user 300 hey, thinking about you`);
            console.log("");
            continue;
          }
          const target = args[0];
          const delay = parseInt(args[1]) || 0;
          const msg = args.slice(2).join(" ");
          try {
            const params = new URLSearchParams({ message: msg, delay_seconds: String(delay) });
            await apiCall<any>(`/intervene/${target}/queue?${params}`, "POST");
            console.log(`  ${pc.cyan("📬")} Queued → ${pc.white(target)} in ${pc.yellow(String(delay))}s: "${pc.dim(msg.slice(0, 50))}"`);
          } catch {
            console.log(`  ${pc.red("!")} ${pc.dim("Failed to queue message")}`);
          }
          console.log("");
          continue;
        }

        case "/intervene": {
          try {
            const data = await apiCall<any>("/intervene/status");
            console.log("");
            console.log(DIVIDER);
            console.log(pc.bold(pc.white("  ACTIVE INTERVENTIONS")));
            console.log(DIVIDER);
            console.log("");

            const paused = Object.entries(data.paused_chats || {});
            const interventions = Object.entries(data.active_interventions || {});
            const overrides = Object.entries(data.next_reply_overrides || {});

            if (paused.length === 0 && interventions.length === 0 && overrides.length === 0) {
              console.log(`  ${pc.dim("No active interventions")}`);
            } else {
              if (paused.length > 0) {
                console.log(`  ${pc.bold(pc.yellow("PAUSED CHATS"))}`);
                paused.forEach(([cid, info]: any) => {
                  console.log(`  ${pc.yellow("⏸")} ${pc.white(cid)} — resumes in ${pc.cyan(String(Math.round(info.resumes_in_min)))}min (${info.resumes_at})`);
                });
                console.log("");
              }
              if (interventions.length > 0) {
                console.log(`  ${pc.bold(pc.magenta("ACTIVE INSTRUCTIONS"))}`);
                interventions.forEach(([cid, info]: any) => {
                  console.log(`  ${pc.magenta("🎯")} ${pc.white(cid)} — "${pc.cyan(info.instruction)}" (${Math.round(info.expires_in_min)}min left)`);
                });
                console.log("");
              }
              if (overrides.length > 0) {
                console.log(`  ${pc.bold(pc.blue("ONE-SHOT OVERRIDES"))}`);
                overrides.forEach(([cid, inst]: any) => {
                  console.log(`  ${pc.blue("→")} ${pc.white(cid)} — "${pc.cyan(inst)}"`);
                });
                console.log("");
              }
            }
            console.log(DIVIDER);
            console.log("");
          } catch {
            console.log(`  ${pc.red("!")} ${pc.dim("Interventions unavailable")}`);
            console.log("");
          }
          continue;
        }

        // ─── VOICE COMMANDS ──────────────────────────────────
        case "/voice": {
          const subCmd = parts[1]?.toLowerCase();
          if (!subCmd || subCmd === "status") {
            // Show voice status
            try {
              const data = await apiCall<any>("/voice/status");
              console.log("");
              console.log(DIVIDER);
              console.log(pc.bold(pc.white("  VOICE ENGINE")));
              console.log(DIVIDER);
              console.log("");

              const backends = data.backends || {};
              Object.entries(backends).forEach(([name, info]: any) => {
                const status = info.available ? pc.green("●") : pc.dim("○");
                const cloning = info.voice_cloning ? pc.green(" [cloning]") : "";
                console.log(`  ${status} ${pc.white(name.padEnd(14))} ${pc.dim(info.quality || "")}${cloning}`);
              });

              console.log("");
              const vc = data.voice_cloning || {};
              const voiceReg = vc.user_voice_registered ? pc.green("registered") : pc.red("not registered");
              console.log(`  ${pc.dim("Your voice:")}  ${voiceReg}`);
              if (vc.named_voices > 0) console.log(`  ${pc.dim("Personas:")}    ${pc.cyan(String(vc.named_voices))}`);
              if (vc.chat_specific_voices > 0) console.log(`  ${pc.dim("Chat voices:")} ${pc.cyan(String(vc.chat_specific_voices))}`);
              console.log(`  ${pc.dim("Cached:")}      ${pc.cyan(String(data.cached_files || 0))} files`);

              console.log("");
              console.log(`  ${pc.dim("Register:")} ${pc.cyan("/voice register saved")} ${pc.dim("(from Saved Messages)")}`);
              console.log(`  ${pc.dim("Send:")}     ${pc.cyan("/voice send @user <text>")}`);
              console.log("");
              console.log(DIVIDER);
              console.log("");
            } catch {
              console.log(`  ${pc.red("!")} ${pc.dim("Voice engine unavailable")}`);
              console.log("");
            }
            continue;
          } else if (subCmd === "register") {
            const source = parts[2] || "saved";
            passToAgent = true;
            agentOverride = `Register my voice for cloning from ${source === "saved" ? "Saved Messages" : source}. Use the registerVoice tool with chat_id="${source === "saved" ? "saved" : source}".`;
            break;
          } else if (subCmd === "send") {
            const target = parts[2];
            const text = parts.slice(3).join(" ");
            if (!target || !text) {
              console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/voice send @user <text>")}`);
              console.log("");
              continue;
            }
            passToAgent = true;
            agentOverride = `Send a voice message to ${target} saying: "${text}". Use the sendVoiceClone tool.`;
            break;
          } else if (subCmd === "list") {
            try {
              const data = await apiCall<any>("/voice/voices");
              console.log("");
              console.log(DIVIDER);
              console.log(pc.bold(pc.white("  AVAILABLE VOICES")));
              console.log(DIVIDER);
              console.log("");

              const myVoice = data.my_voice || [];
              if (myVoice.length > 0) {
                console.log(`  ${pc.bold(pc.cyan("MY VOICE"))}`);
                myVoice.forEach((v: any) => {
                  console.log(`  ${pc.green("●")} ${pc.white(v.name)} ${pc.dim(`(${v.format}, ${v.size_kb}KB)`)} ${pc.dim(v.path)}`);
                });
                console.log("");
              }

              const named = data.named_voices || [];
              if (named.length > 0) {
                console.log(`  ${pc.bold(pc.magenta("NAMED PERSONAS"))}`);
                named.forEach((v: any) => {
                  console.log(`  ${pc.magenta("●")} ${pc.white(v.name)} ${pc.dim(`(${v.format})`)} ${pc.dim(v.path)}`);
                });
                console.log("");
              }

              const chatVoices = data.chat_voices || {};
              const chatEntries = Object.entries(chatVoices);
              if (chatEntries.length > 0) {
                console.log(`  ${pc.bold(pc.blue("CHAT REFERENCES"))}`);
                chatEntries.forEach(([cid, files]: any) => {
                  console.log(`  ${pc.blue("●")} Chat ${pc.white(cid)}: ${pc.dim(`${files.length} samples`)}`);
                });
                console.log("");
              }

              if (myVoice.length === 0 && named.length === 0 && chatEntries.length === 0) {
                console.log(`  ${pc.dim("No voices registered yet")}`);
                console.log(`  ${pc.dim("Register:")} ${pc.cyan("/voice register saved")}`);
                console.log("");
              }

              console.log(DIVIDER);
              console.log("");
            } catch {
              console.log(`  ${pc.red("!")} ${pc.dim("Voice listing unavailable")}`);
              console.log("");
            }
            continue;
          } else if (subCmd === "assign") {
            const target = parts[2];
            const voicePath = parts.slice(3).join(" ");
            if (!target || !voicePath) {
              console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/voice assign @user <voice_path>")}`);
              console.log("");
              continue;
            }
            passToAgent = true;
            agentOverride = `Assign the voice at path "${voicePath}" to chat ${target}. Use the assignVoice tool.`;
            break;
          } else {
            console.log(`  ${pc.yellow("?")} ${pc.dim("Voice subcommands:")} register, send, list, assign, status`);
            console.log("");
            continue;
          }
        }

        // ─── CALL COMMANDS ─────────────────────────────────────
        case "/call": {
          const subCmd = parts[1]?.toLowerCase();
          if (!subCmd || subCmd === "status") {
            // Show call status
            try {
              const data = await apiCall<any>("/call/status");
              console.log("");
              console.log(DIVIDER);
              console.log(pc.bold(pc.white("  CALL ENGINE")));
              console.log(DIVIDER);
              console.log("");

              const avail = data.available ? pc.green("available") : pc.red("unavailable");
              const bridge = data.bridge?.ok ? pc.green("running") : pc.yellow("stopped");
              console.log(`  ${pc.dim("Backend:")}     ${avail} ${pc.dim(`(${data.backend || "none"})`)}`);
              console.log(`  ${pc.dim("Bridge:")}      ${bridge}`);
              console.log(`  ${pc.dim("Private:")}     ${data.private_calls ? pc.green("yes") : pc.red("no")}`);
              console.log(`  ${pc.dim("Group:")}       ${data.group_calls ? pc.green("yes") : pc.red("no")}`);

              const calls = data.calls || {};
              const callCount = Object.keys(calls).length;
              if (callCount > 0) {
                console.log("");
                console.log(pc.bold(pc.cyan("  ACTIVE CALLS")));
                Object.entries(calls).forEach(([cid, info]: any) => {
                  const icon = info.call_type === "group" ? "🔊" : "📞";
                  console.log(`  ${icon} ${pc.white(cid)} ${pc.dim(`${info.status} | ${info.duration_s}s | ${info.direction}`)}`);
                });
              } else {
                console.log(`  ${pc.dim("No active calls")}`);
              }

              console.log("");
              console.log(`  ${pc.dim("Call:")}   ${pc.cyan("/call @user")} ${pc.dim("(private call)")}`);
              console.log(`  ${pc.dim("Group:")}  ${pc.cyan("/call group @chat")} ${pc.dim("(join voice chat)")}`);
              console.log(`  ${pc.dim("Speak:")}  ${pc.cyan("/call speak @user <text>")}`);
              console.log("");
              console.log(DIVIDER);
              console.log("");
            } catch {
              console.log(`  ${pc.red("!")} ${pc.dim("Call engine unavailable")}`);
              console.log("");
            }
            continue;
          } else if (subCmd === "start" || subCmd === "bridge") {
            try {
              console.log(`  ${pc.yellow("◐")} ${pc.dim("Starting call bridge...")}`);
              const data = await apiCall<any>("/call/start-bridge", "POST");
              if (data.success) {
                console.log(`  ${pc.green("●")} ${pc.white("Call bridge started")}`);
              } else {
                console.log(`  ${pc.red("!")} ${data.error || "Failed to start bridge"}`);
              }
              console.log("");
            } catch {
              console.log(`  ${pc.red("!")} ${pc.dim("Failed to start call bridge")}`);
              console.log("");
            }
            continue;
          } else if (subCmd === "answer" || subCmd === "accept") {
            const target = parts[2];
            if (!target) {
              console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/call answer @user")}`);
              console.log("");
              continue;
            }
            try {
              console.log(`  ${pc.yellow("◐")} ${pc.dim("Accepting call from")} ${pc.cyan(target)}${pc.dim("...")}`);
              const data = await apiCall<any>(`/call/accept/${encodeURIComponent(target)}`, "POST");
              if (data.success) {
                console.log(`  ${pc.green("●")} ${pc.white("Call accepted")} ${pc.dim("with")} ${pc.cyan(target)}`);
              } else {
                console.log(`  ${pc.red("!")} ${data.error || "Failed to accept call"}`);
              }
            } catch (e: any) {
              console.log(`  ${pc.red("!")} ${pc.dim(e.message || "Failed to accept call")}`);
            }
            console.log("");
            continue;
          } else if (subCmd === "decline" || subCmd === "reject") {
            const target = parts[2];
            if (!target) {
              console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/call decline @user")}`);
              console.log("");
              continue;
            }
            try {
              const data = await apiCall<any>(`/call/decline/${encodeURIComponent(target)}`, "POST");
              if (data.success) {
                console.log(`  ${pc.red("●")} ${pc.white("Call declined")} ${pc.dim("from")} ${pc.cyan(target)}`);
              } else {
                console.log(`  ${pc.red("!")} ${data.error || "Failed to decline call"}`);
              }
            } catch (e: any) {
              console.log(`  ${pc.red("!")} ${pc.dim(e.message || "Failed to decline call")}`);
            }
            console.log("");
            continue;
          } else if (subCmd === "hangup" || subCmd === "end") {
            const target = parts[2];
            if (!target) {
              console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/call hangup @user")}`);
              console.log("");
              continue;
            }
            try {
              const data = await apiCall<any>(`/call/hangup/${encodeURIComponent(target)}`, "POST");
              if (data.success) {
                const dur = data.duration_s ? ` ${pc.dim(`(${data.duration_s}s)`)}` : "";
                console.log(`  ${pc.red("●")} ${pc.white("Call ended")} ${pc.dim("with")} ${pc.cyan(target)}${dur}`);
              } else {
                console.log(`  ${pc.red("!")} ${data.error || "Failed to hang up"}`);
              }
            } catch (e: any) {
              console.log(`  ${pc.red("!")} ${pc.dim(e.message || "Failed to hang up")}`);
            }
            console.log("");
            continue;
          } else if (subCmd === "speak" || subCmd === "say") {
            const target = parts[2];
            const text = parts.slice(3).join(" ");
            if (!target || !text) {
              console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/call speak @user <text>")}`);
              console.log("");
              continue;
            }
            try {
              const params = new URLSearchParams({ text });
              console.log(`  ${pc.yellow("◐")} ${pc.dim("Speaking...")} ${pc.white(`"${text.slice(0, 50)}"`)}`);
              const data = await apiCall<any>(`/call/speak/${encodeURIComponent(target)}?${params.toString()}`, "POST");
              if (data.success) {
                console.log(`  ${pc.green("●")} ${pc.white("Spoken in call")} ${pc.dim("with")} ${pc.cyan(target)}`);
              } else {
                console.log(`  ${pc.red("!")} ${data.error || "Failed to speak"}`);
              }
            } catch (e: any) {
              console.log(`  ${pc.red("!")} ${pc.dim(e.message || "Failed to speak in call")}`);
            }
            console.log("");
            continue;
          } else if (subCmd === "listen") {
            const target = parts[2];
            if (!target) {
              console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/call listen @user")}`);
              console.log("");
              continue;
            }
            try {
              console.log(`  ${pc.yellow("◐")} ${pc.dim("Listening...")}`);
              const data = await apiCall<any>(`/call/listen/${encodeURIComponent(target)}`);
              if (data.success) {
                if (data.text) {
                  console.log(`  ${pc.green("●")} ${pc.dim("They said:")} ${pc.white(`"${data.text}"`)}`);
                  if (data.audio_duration_s) {
                    console.log(`  ${pc.dim(`  (${data.audio_duration_s}s of audio)`)}`);
                  }
                } else {
                  console.log(`  ${pc.dim("  No speech detected yet")}`);
                }
              } else {
                console.log(`  ${pc.red("!")} ${data.error || "Failed to listen"}`);
              }
            } catch (e: any) {
              console.log(`  ${pc.red("!")} ${pc.dim(e.message || "Failed to listen")}`);
            }
            console.log("");
            continue;
          } else if (subCmd === "auto" || subCmd === "autonomy") {
            const arg2 = parts[2]?.toLowerCase();
            if (arg2 === "off" || arg2 === "disable") {
              const target = parts[3];
              if (!target) {
                console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/call auto off @user")}`);
                console.log("");
                continue;
              }
              try {
                const params = new URLSearchParams({ enabled: "false" });
                const data = await apiCall<any>(`/call/autonomy/${encodeURIComponent(target)}?${params.toString()}`, "POST");
                if (data.success) {
                  console.log(`  ${pc.red("●")} ${pc.white("Autonomy OFF")} ${pc.dim("for")} ${pc.cyan(target)}`);
                } else {
                  console.log(`  ${pc.red("!")} ${data.error || "Failed to disable autonomy"}`);
                }
              } catch (e: any) {
                console.log(`  ${pc.red("!")} ${pc.dim(e.message || "Failed")}`);
              }
              console.log("");
              continue;
            } else if (arg2) {
              // /call auto @user [language]
              const target = arg2;
              const language = parts[3] || "auto";
              try {
                const params = new URLSearchParams({ enabled: "true", language });
                console.log(`  ${pc.yellow("◐")} ${pc.dim("Enabling autonomy for")} ${pc.cyan(target)} ${pc.dim(`(${language})...`)}`);
                const data = await apiCall<any>(`/call/autonomy/${encodeURIComponent(target)}?${params.toString()}`, "POST");
                if (data.success) {
                  console.log(`  ${pc.green("●")} ${pc.white("Autonomy ON")} ${pc.dim("for")} ${pc.cyan(target)} ${pc.dim(`(${language})`)}`);
                  console.log(`  ${pc.dim("  Bot is now listening, thinking, and speaking on its own")}`);
                } else {
                  console.log(`  ${pc.red("!")} ${data.error || "Failed to enable autonomy"}`);
                }
              } catch (e: any) {
                console.log(`  ${pc.red("!")} ${pc.dim(e.message || "Failed")}`);
              }
              console.log("");
              continue;
            } else {
              console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")}`);
              console.log(`    ${pc.cyan("/call auto @user")}        ${pc.dim("Enable autonomy")}`);
              console.log(`    ${pc.cyan("/call auto @user ru")}     ${pc.dim("Enable with language")}`);
              console.log(`    ${pc.cyan("/call auto off @user")}    ${pc.dim("Disable autonomy")}`);
              console.log("");
              continue;
            }
          } else if (subCmd === "autoanswer" || subCmd === "auto-answer") {
            const toggle = parts[2]?.toLowerCase();
            if (toggle === "on" || toggle === "true" || toggle === "enable") {
              const withAutonomy = parts[3]?.toLowerCase() === "auto";
              try {
                const params = new URLSearchParams({
                  enabled: "true",
                  with_autonomy: withAutonomy ? "true" : "false",
                });
                await apiCall<any>(`/call/auto-accept?${params.toString()}`, "POST");
                console.log(`  ${pc.green("●")} ${pc.white("Auto-accept:")} ${pc.green("ON")}${withAutonomy ? ` ${pc.cyan("+ autonomy")}` : ""}`);
              } catch {
                console.log(`  ${pc.red("!")} ${pc.dim("Failed to set auto-accept")}`);
              }
              console.log("");
              continue;
            } else if (toggle === "off" || toggle === "false" || toggle === "disable") {
              try {
                await apiCall<any>("/call/auto-accept?enabled=false", "POST");
                console.log(`  ${pc.green("●")} ${pc.white("Auto-accept:")} ${pc.red("OFF")}`);
              } catch {
                console.log(`  ${pc.red("!")} ${pc.dim("Failed to set auto-accept")}`);
              }
              console.log("");
              continue;
            } else {
              console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")}`);
              console.log(`    ${pc.cyan("/call autoanswer on")}       ${pc.dim("Auto-accept calls")}`);
              console.log(`    ${pc.cyan("/call autoanswer on auto")}  ${pc.dim("Auto-accept + full autonomy")}`);
              console.log(`    ${pc.cyan("/call autoanswer off")}      ${pc.dim("Disable auto-accept")}`);
              console.log("");
              continue;
            }
          } else if (subCmd === "group") {
            const action = parts[2]?.toLowerCase();
            const target = parts[3];
            if (!action || !target) {
              console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/call group join @chat")}`);
              console.log(`  ${pc.dim("        ")} ${pc.cyan("/call group leave @chat")}`);
              console.log(`  ${pc.dim("        ")} ${pc.cyan("/call group speak @chat <text>")}`);
              console.log("");
              continue;
            }
            if (action === "join") {
              try {
                console.log(`  ${pc.yellow("◐")} ${pc.dim("Joining voice chat in")} ${pc.cyan(target)}${pc.dim("...")}`);
                const data = await apiCall<any>(`/call/group/join/${encodeURIComponent(target)}`, "POST");
                if (data.success) {
                  console.log(`  ${pc.green("●")} ${pc.white("Joined group call")} ${pc.dim("in")} ${pc.cyan(target)}`);
                } else {
                  console.log(`  ${pc.red("!")} ${data.error || "Failed to join"}`);
                }
              } catch (e: any) {
                console.log(`  ${pc.red("!")} ${pc.dim(e.message || "Failed to join group call")}`);
              }
              console.log("");
              continue;
            } else if (action === "leave") {
              try {
                const data = await apiCall<any>(`/call/group/leave/${encodeURIComponent(target)}`, "POST");
                if (data.success) {
                  console.log(`  ${pc.red("●")} ${pc.white("Left group call")} ${pc.dim("in")} ${pc.cyan(target)}`);
                } else {
                  console.log(`  ${pc.red("!")} ${data.error || "Failed to leave"}`);
                }
              } catch (e: any) {
                console.log(`  ${pc.red("!")} ${pc.dim(e.message || "Failed to leave group call")}`);
              }
              console.log("");
              continue;
            } else if (action === "speak" || action === "say") {
              const text = parts.slice(4).join(" ");
              if (!text) {
                console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/call group speak @chat <text>")}`);
                console.log("");
                continue;
              }
              try {
                const params = new URLSearchParams({ text });
                console.log(`  ${pc.yellow("◐")} ${pc.dim("Speaking in group...")} ${pc.white(`"${text.slice(0, 50)}"`)}`);
                const data = await apiCall<any>(`/call/group/speak/${encodeURIComponent(target)}?${params.toString()}`, "POST");
                if (data.success) {
                  console.log(`  ${pc.green("●")} ${pc.white("Spoken in group")} ${pc.cyan(target)}`);
                } else {
                  console.log(`  ${pc.red("!")} ${data.error || "Failed to speak in group"}`);
                }
              } catch (e: any) {
                console.log(`  ${pc.red("!")} ${pc.dim(e.message || "Failed")}`);
              }
              console.log("");
              continue;
            }
            continue;
          } else if (subCmd === "setup") {
            // One-command setup for the call bridge venv
            console.log("");
            console.log(DIVIDER);
            console.log(pc.bold(pc.white("  CALL BRIDGE SETUP")));
            console.log(DIVIDER);
            console.log("");
            console.log(`  ${pc.dim("Run these commands in your project root:")}`);
            console.log("");
            console.log(`  ${pc.cyan("1.")} ${pc.white("uv venv .venv-calls --python 3.10")}`);
            console.log(`  ${pc.cyan("2.")} ${pc.white("uv pip install --python .venv-calls/bin/python tgcalls==2.0.0 pytgcalls==2.1.0 telethon==1.42.0 python-dotenv pyaes pyasn1 rsa")}`);
            console.log(`  ${pc.cyan("3.")} ${pc.white("/call bridge")} ${pc.dim("(starts the bridge)")}`);
            console.log("");
            console.log(`  ${pc.dim("Then you can make calls:")} ${pc.cyan("/call @username")}`);
            console.log("");
            console.log(DIVIDER);
            console.log("");
            continue;
          } else {
            // Default: treat as /call @user (make a call)
            const target = subCmd;
            const initialMsg = parts.slice(2).join(" ");
            try {
              console.log(`  ${pc.yellow("◐")} ${pc.dim("Calling")} ${pc.cyan(target)}${pc.dim("...")}`);
              const params = initialMsg ? `?initial_message=${encodeURIComponent(initialMsg)}` : "";
              const data = await apiCall<any>(`/call/make/${encodeURIComponent(target)}${params}`, "POST");
              if (data.success) {
                console.log(`  ${pc.green("●")} ${pc.white("Call started")} ${pc.dim("with")} ${pc.cyan(target)}`);
                if (data.call_type) console.log(`  ${pc.dim(`  Type: ${data.call_type}`)}`);
                if (initialMsg) console.log(`  ${pc.dim(`  Said: "${initialMsg.slice(0, 40)}"`)}`);
              } else {
                console.log(`  ${pc.red("!")} ${data.error || "Failed to make call"}`);
              }
            } catch (e: any) {
              console.log(`  ${pc.red("!")} ${pc.dim(e.message || "Failed to make call")}`);
            }
            console.log("");
            continue;
          }
        }

        // ─── INTELLIGENCE COMMANDS ──────────────────────────
        // Agent-routed commands
        case "/analyze": {
          const target = parts.slice(1).join(" ");
          if (!target) {
            console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/analyze @username")}`);
            console.log("");
            continue;
          }
          passToAgent = true;
          agentOverride = `Run a comprehensive V5 psychological analysis on ${target}. Give me the key insights, conversation health, emotional state, and any warnings.`;
          break;
        }

        case "/health": {
          const target = parts.slice(1).join(" ");
          if (!target) {
            console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/health @username")}`);
            console.log("");
            continue;
          }
          passToAgent = true;
          agentOverride = `Get the conversation health score for ${target}. Show me the breakdown of all signals.`;
          break;
        }

        case "/memory": {
          const target = parts.slice(1).join(" ");
          if (!target) {
            console.log(`  ${pc.yellow("?")} ${pc.dim("Usage:")} ${pc.cyan("/memory @username")}`);
            console.log("");
            continue;
          }
          passToAgent = true;
          agentOverride = `Show me everything the bot remembers about ${target}. Use both getAdvancedMemory and getChatMemory to get the full picture.`;
          break;
        }

        case "/train":
          passToAgent = true;
          agentOverride = "Trigger ML model training for all tasks with include_neural=true. Show me the results when done.";
          break;

        case "/quit":
        case "/exit":
        case "/q":
          console.log("");
          console.log(THIN_DIVIDER);
          console.log(`  ${pc.dim("See you later")} ${pc.magenta("♥")}`);
          console.log(THIN_DIVIDER);
          console.log("");
          process.exit(0);

        default:
          console.log(`  ${pc.yellow("?")} ${pc.dim("Unknown command:")} ${cmd} ${pc.dim("· try /help")}`);
          console.log("");
          continue;
      }

      if (!passToAgent) continue;
    }

    // Process with AI agent
    const actualMessage = agentOverride || message;

    const spinner = p.spinner();
    spinner.start(pc.dim("thinking..."));

    try {
      const stream = await chat(actualMessage);
      spinner.stop(`${formatTimestamp()} ${pc.bold(pc.magenta("agent"))}`);

      // Stream the response with nice formatting
      let response = "";
      process.stdout.write(pc.dim("  "));

      for await (const chunk of stream) {
        // Add indentation after newlines for clean formatting
        const formatted = chunk.replace(/\n/g, `\n  `);
        process.stdout.write(formatted);
        response += chunk;
      }

      console.log("");
      console.log("");
    } catch (error: any) {
      spinner.stop(`${pc.red("  ✕ Error")}`);

      if (error.message?.includes("Telegram API") || error.message?.includes("fetch")) {
        console.log(
          `  ${pc.red("!")} ${pc.dim("Telegram API error. Is the bridge running?")}`
        );
        console.log(`  ${pc.dim("$")} ${pc.cyan("uv run python telegram_api.py")}`);
      } else if (error.message?.includes("rate") || error.message?.includes("429")) {
        console.log(`  ${pc.yellow("!")} ${pc.dim("Rate limited. Wait a moment and try again.")}`);
      } else {
        console.log(`  ${pc.red("!")} ${pc.dim(error.message || "Something went wrong")}`);
      }
      console.log("");
    }
  }
}

// Run
main().catch((error) => {
  console.error(`\n  ${pc.red("Fatal:")} ${error.message}`);
  process.exit(1);
});
