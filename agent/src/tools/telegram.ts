/**
 * Telegram tools for the AI agent
 * Calls the Python HTTP bridge to interact with Telegram
 */

import { tool } from "ai";
import { z } from "zod";
import { config } from "../config";

const API = config.telegramApiUrl;

// Helper for API calls
async function telegramFetch<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Telegram API error: ${error}`);
  }

  return response.json();
}

// Types
interface Chat {
  id: number;
  type: string;
  first_name?: string;
  last_name?: string;
  title?: string;
  username?: string;
  unread_count?: number;
  last_message?: string;
}

interface Message {
  id: number;
  date: string;
  text: string;
  out: boolean;
  sender_name: string;
  sender_id: number;
  reply_to_msg_id?: number;
  has_media: boolean;
  media_type?: string;
}

// Tools

export const getChats = tool({
  description: `List all Telegram chats (conversations). Returns chat ID, name, type, and last message preview. Use this to find someone's chat ID before reading or sending messages.`,
  inputSchema: z.object({
    limit: z.number().min(1).max(100).default(30).describe("Number of chats to return"),
    chat_type: z
      .enum(["user", "chat", "channel"])
      .optional()
      .describe("Filter by type: 'user' for DMs, 'chat' for groups, 'channel' for channels"),
  }),
  execute: async ({ limit, chat_type }) => {
    const params = new URLSearchParams();
    params.set("limit", String(limit));
    if (chat_type) params.set("chat_type", chat_type);

    const data = await telegramFetch<{ chats: Chat[]; count: number }>(
      `/chats?${params.toString()}`
    );

    // Format for better readability
    const formatted = data.chats.map((chat) => {
      const name = chat.title || `${chat.first_name || ""} ${chat.last_name || ""}`.trim() || "Unknown";
      return {
        id: chat.id,
        name,
        type: chat.type,
        username: chat.username,
        unread: chat.unread_count || 0,
        preview: chat.last_message?.slice(0, 50),
      };
    });

    return { chats: formatted, total: data.count };
  },
});

export const getMessages = tool({
  description: `Read messages from a specific Telegram chat. Returns message ID, text, sender, date. Use after getChats to get the chat_id.`,
  inputSchema: z.object({
    chat_id: z
      .union([z.number(), z.string()])
      .describe("Chat ID (number) or username (string like '@username')"),
    limit: z.number().min(1).max(50).default(10).describe("Number of messages to fetch"),
    offset_id: z.number().optional().describe("Get messages before this message ID (for pagination)"),
  }),
  execute: async ({ chat_id, limit, offset_id }) => {
    const params = new URLSearchParams();
    params.set("limit", String(limit));
    if (offset_id) params.set("offset_id", String(offset_id));

    const data = await telegramFetch<{ messages: Message[]; count: number }>(
      `/chats/${chat_id}/messages?${params.toString()}`
    );

    // Format messages for readability
    const formatted = data.messages.map((msg) => ({
      id: msg.id,
      from: msg.sender_name,
      text: msg.text,
      date: msg.date,
      isFromMe: msg.out,
      hasMedia: msg.has_media,
      mediaType: msg.media_type,
      replyTo: msg.reply_to_msg_id,
    }));

    return { messages: formatted, count: data.count };
  },
});

export const sendMessage = tool({
  description: `Send a text message to a Telegram chat. Returns success status and message ID.`,
  inputSchema: z.object({
    chat_id: z
      .union([z.number(), z.string()])
      .describe("Chat ID (number) or username (string like '@username')"),
    message: z.string().min(1).max(4096).describe("Message text to send"),
    reply_to: z.number().optional().describe("Message ID to reply to (optional)"),
  }),
  execute: async ({ chat_id, message, reply_to }) => {
    const data = await telegramFetch<{ success: boolean; message_id: number; date: string }>(
      `/chats/${chat_id}/messages`,
      {
        method: "POST",
        body: JSON.stringify({ message, reply_to }),
      }
    );

    return {
      success: data.success,
      messageId: data.message_id,
      sentAt: data.date,
    };
  },
});

export const scheduleMessage = tool({
  description: `Schedule a message to be sent at a future time. Perfect for sending good morning/night messages.`,
  inputSchema: z.object({
    chat_id: z
      .union([z.number(), z.string()])
      .describe("Chat ID (number) or username (string like '@username')"),
    message: z.string().min(1).max(4096).describe("Message text to send"),
    minutes_from_now: z.number().min(1).max(525600).describe("Minutes from now to send (1-525600, max 1 year)"),
  }),
  execute: async ({ chat_id, message, minutes_from_now }) => {
    const data = await telegramFetch<{ success: boolean; message_id: number; scheduled_for: string }>(
      `/chats/${chat_id}/schedule`,
      {
        method: "POST",
        body: JSON.stringify({ message, minutes_from_now }),
      }
    );

    return {
      success: data.success,
      messageId: data.message_id,
      scheduledFor: data.scheduled_for,
    };
  },
});

export const getChat = tool({
  description: `Get detailed information about a specific chat by ID or username.`,
  inputSchema: z.object({
    chat_id: z
      .union([z.number(), z.string()])
      .describe("Chat ID (number) or username (string like '@username')"),
  }),
  execute: async ({ chat_id }) => {
    const data = await telegramFetch<Chat>(`/chats/${chat_id}`);
    return data;
  },
});

export const searchContacts = tool({
  description: `Search for contacts by name, username, or phone number.`,
  inputSchema: z.object({
    query: z.string().min(1).describe("Search query (name, username, or phone)"),
  }),
  execute: async ({ query }) => {
    const params = new URLSearchParams({ query });
    const data = await telegramFetch<{ contacts: any[]; count: number }>(
      `/contacts/search?${params.toString()}`
    );
    return data;
  },
});

// ============= NEW TOOLS =============

export const getHistory = tool({
  description: `Get full chat history (up to 500 messages). Use for getting more context about the conversation.`,
  inputSchema: z.object({
    chat_id: z.union([z.number(), z.string()]).describe("Chat ID or username"),
    limit: z.number().min(1).max(500).default(100).describe("Number of messages"),
  }),
  execute: async ({ chat_id, limit }) => {
    const params = new URLSearchParams({ limit: String(limit) });
    const data = await telegramFetch<{ messages: Message[]; count: number }>(
      `/chats/${chat_id}/history?${params.toString()}`
    );
    return {
      messages: data.messages.map((msg) => ({
        id: msg.id,
        from: msg.sender_name,
        text: msg.text,
        date: msg.date,
        isFromMe: msg.out,
      })),
      count: data.count,
    };
  },
});

export const sendReaction = tool({
  description: `Send a reaction emoji to a message. Perfect for reacting to her messages with ❤️ 🔥 😂 😮 😢 🎉 👍 👎`,
  inputSchema: z.object({
    chat_id: z.union([z.number(), z.string()]).describe("Chat ID or username"),
    message_id: z.number().describe("Message ID to react to"),
    emoji: z.string().describe("Emoji to react with (e.g., '❤️', '🔥', '😂')"),
    big: z.boolean().default(false).describe("Show big animation"),
  }),
  execute: async ({ chat_id, message_id, emoji, big }) => {
    const data = await telegramFetch<{ success: boolean; emoji: string }>(
      `/chats/${chat_id}/messages/${message_id}/reaction`,
      {
        method: "POST",
        body: JSON.stringify({ emoji, big }),
      }
    );
    return data;
  },
});

export const replyToMessage = tool({
  description: `Reply directly to a specific message. Creates a reply thread.`,
  inputSchema: z.object({
    chat_id: z.union([z.number(), z.string()]).describe("Chat ID or username"),
    message_id: z.number().describe("Message ID to reply to"),
    message: z.string().min(1).describe("Reply text"),
  }),
  execute: async ({ chat_id, message_id, message }) => {
    const data = await telegramFetch<{ success: boolean; message_id: number }>(
      `/chats/${chat_id}/messages/${message_id}/reply`,
      {
        method: "POST",
        body: JSON.stringify({ message }),
      }
    );
    return data;
  },
});

export const editMessage = tool({
  description: `Edit a message you sent. Fix typos or update content.`,
  inputSchema: z.object({
    chat_id: z.union([z.number(), z.string()]).describe("Chat ID or username"),
    message_id: z.number().describe("Message ID to edit"),
    new_text: z.string().min(1).describe("New message text"),
  }),
  execute: async ({ chat_id, message_id, new_text }) => {
    const data = await telegramFetch<{ success: boolean }>(
      `/chats/${chat_id}/messages/${message_id}`,
      {
        method: "PUT",
        body: JSON.stringify({ new_text }),
      }
    );
    return data;
  },
});

export const deleteMessage = tool({
  description: `Delete a message. Use to remove embarrassing messages.`,
  inputSchema: z.object({
    chat_id: z.union([z.number(), z.string()]).describe("Chat ID or username"),
    message_id: z.number().describe("Message ID to delete"),
  }),
  execute: async ({ chat_id, message_id }) => {
    const data = await telegramFetch<{ success: boolean }>(
      `/chats/${chat_id}/messages/${message_id}`,
      { method: "DELETE" }
    );
    return data;
  },
});

export const forwardMessage = tool({
  description: `Forward a message to another chat. Great for sharing memes or content.`,
  inputSchema: z.object({
    chat_id: z.union([z.number(), z.string()]).describe("Source chat ID"),
    message_id: z.number().describe("Message ID to forward"),
    to_chat_id: z.union([z.number(), z.string()]).describe("Destination chat ID"),
  }),
  execute: async ({ chat_id, message_id, to_chat_id }) => {
    const params = new URLSearchParams({ to_chat_id: String(to_chat_id) });
    const data = await telegramFetch<{ success: boolean }>(
      `/chats/${chat_id}/messages/${message_id}/forward?${params.toString()}`,
      { method: "POST" }
    );
    return data;
  },
});

export const markAsRead = tool({
  description: `Mark all messages in a chat as read.`,
  inputSchema: z.object({
    chat_id: z.union([z.number(), z.string()]).describe("Chat ID or username"),
  }),
  execute: async ({ chat_id }) => {
    const data = await telegramFetch<{ success: boolean }>(
      `/chats/${chat_id}/read`,
      { method: "POST" }
    );
    return data;
  },
});

export const pinMessage = tool({
  description: `Pin an important message in the chat.`,
  inputSchema: z.object({
    chat_id: z.union([z.number(), z.string()]).describe("Chat ID or username"),
    message_id: z.number().describe("Message ID to pin"),
  }),
  execute: async ({ chat_id, message_id }) => {
    const data = await telegramFetch<{ success: boolean }>(
      `/chats/${chat_id}/messages/${message_id}/pin`,
      { method: "POST" }
    );
    return data;
  },
});

export const searchMessages = tool({
  description: `Search for messages in a chat by text. Find specific conversations.`,
  inputSchema: z.object({
    chat_id: z.union([z.number(), z.string()]).describe("Chat ID or username"),
    query: z.string().min(1).describe("Search text"),
    limit: z.number().min(1).max(100).default(20).describe("Max results"),
  }),
  execute: async ({ chat_id, query, limit }) => {
    const params = new URLSearchParams({ query, limit: String(limit) });
    const data = await telegramFetch<{ messages: Message[]; count: number }>(
      `/chats/${chat_id}/search?${params.toString()}`
    );
    return {
      messages: data.messages.map((msg) => ({
        id: msg.id,
        from: msg.sender_name,
        text: msg.text,
        date: msg.date,
      })),
      count: data.count,
    };
  },
});

export const getUserStatus = tool({
  description: `Check if a user is online. See when she was last active.`,
  inputSchema: z.object({
    user_id: z.union([z.number(), z.string()]).describe("User ID or username"),
  }),
  execute: async ({ user_id }) => {
    const data = await telegramFetch<{ user_id: number; status: string; raw_status: string }>(
      `/users/${user_id}/status`
    );
    return data;
  },
});

export const getUserPhotos = tool({
  description: `Get a user's profile photos.`,
  inputSchema: z.object({
    user_id: z.union([z.number(), z.string()]).describe("User ID or username"),
    limit: z.number().min(1).max(50).default(10).describe("Max photos"),
  }),
  execute: async ({ user_id, limit }) => {
    const params = new URLSearchParams({ limit: String(limit) });
    const data = await telegramFetch<{ photos: any[]; count: number }>(
      `/users/${user_id}/photos?${params.toString()}`
    );
    return data;
  },
});

export const searchGifs = tool({
  description: `Search for GIFs to send. Returns a list of available GIFs.`,
  inputSchema: z.object({
    query: z.string().min(1).describe("GIF search query (e.g., 'love', 'funny', 'cute')"),
    limit: z.number().min(1).max(50).default(10).describe("Max results"),
  }),
  execute: async ({ query, limit }) => {
    const params = new URLSearchParams({ query, limit: String(limit) });
    const data = await telegramFetch<{ gifs: any[]; count: number }>(
      `/gifs/search?${params.toString()}`
    );
    return data;
  },
});

// ─── Auto-Reply Management ───────────────────────────────────

export const setAutoReplyInstructions = tool({
  description: `Set custom instructions for how the auto-reply bot should behave in a specific chat. Use this when the user gives tone, style, or language directions for a chat (e.g., "be friendly with @username", "speak russian with @username", "be funny and use lots of emojis with @username"). The instructions persist and the bot follows them autonomously.`,
  inputSchema: z.object({
    chat_id: z.string().describe("The chat username (e.g., '@username') or chat ID"),
    instructions: z.string().describe("Natural language instructions for how the bot should behave in this chat (e.g., 'be friendly, speak russian, use emojis', 'be warm and engaging, ask about their day')"),
  }),
  execute: async ({ chat_id, instructions }) => {
    const data = await telegramFetch<{ success: boolean; chat_id: string; instructions: string }>(
      `/auto-reply/instructions`,
      {
        method: "PUT",
        body: JSON.stringify({ chat_id, instructions }),
      }
    );
    return data;
  },
});

export const getAutoReplyStatus = tool({
  description: `Get the current auto-reply configuration: which chats are active, per-chat instructions, delay settings, and recent reply count. Use this to show the user what's configured.`,
  inputSchema: z.object({}),
  execute: async () => {
    const data = await telegramFetch<any>(`/auto-reply/status`);
    return data;
  },
});

export const toggleAutoReply = tool({
  description: `Enable or disable the autonomous auto-reply system. When enabled, the bot automatically replies to whitelisted chats.`,
  inputSchema: z.object({
    enabled: z.boolean().describe("true to enable auto-reply, false to disable"),
  }),
  execute: async ({ enabled }) => {
    const data = await telegramFetch<{ success: boolean; enabled: boolean }>(
      `/auto-reply/toggle`,
      {
        method: "POST",
        body: JSON.stringify({ enabled }),
      }
    );
    return data;
  },
});

export const getAutoReplyLog = tool({
  description: `Get recent auto-reply activity: what messages were received and what the bot replied. Use this to show the user what the bot has been saying.`,
  inputSchema: z.object({
    limit: z.number().min(1).max(50).default(10).describe("Number of recent entries to return"),
  }),
  execute: async ({ limit }) => {
    const params = new URLSearchParams({ limit: String(limit) });
    const data = await telegramFetch<{ log: any[]; count: number }>(
      `/auto-reply/log?${params.toString()}`
    );
    return data;
  },
});

// ─── NLP & Conversation Intelligence ─────────────────────────

export const analyzeChat = tool({
  description: `Run advanced NLP analysis on a chat. Returns: sentiment analysis (positive/negative/neutral + intensity), conversation stage (new/warming up/flowing/deep/conflict/cooling down/makeup), detected topics, language detection, and a recommended response strategy with tone, length, and priority actions. Use this to understand what's happening in a conversation before responding.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat username (e.g., '@username') or numeric chat ID"),
  }),
  execute: async ({ chat_id }) => {
    const params = new URLSearchParams({ chat_id });
    const data = await telegramFetch<any>(`/nlp/analyze?${params.toString()}`);
    return data;
  },
});

export const getChatMemory = tool({
  description: `Get the conversation memory for a specific chat. Shows: total messages seen, detected language preference, recurring topics, pet names they use, emoji usage patterns, message length patterns, last conflict timestamp, and freeform notes. The bot learns these patterns over time from real conversations.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat username or numeric ID"),
  }),
  execute: async ({ chat_id }) => {
    const params = new URLSearchParams({ chat_id });
    const data = await telegramFetch<any>(`/nlp/memory?${params.toString()}`);
    return data;
  },
});

export const listChatMemories = tool({
  description: `List all stored conversation memories across all chats. Shows a summary of what the bot has learned about each person.`,
  inputSchema: z.object({}),
  execute: async () => {
    const data = await telegramFetch<any>(`/nlp/memories`);
    return data;
  },
});

export const addMemoryNote = tool({
  description: `Add a freeform note to a chat's memory. Use this to teach the bot specific things about a person, like "she likes sunflowers", "her birthday is March 15", "she hates being called babe", "she's studying medicine". These notes persist and influence future auto-replies.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat username or numeric ID"),
    note: z.string().describe("The note to remember (e.g., 'she likes Italian food', 'her cat's name is Milo')"),
  }),
  execute: async ({ chat_id, note }) => {
    const data = await telegramFetch<any>(`/nlp/memory/note`, {
      method: "POST",
      body: JSON.stringify({ chat_id, note }),
    });
    return data;
  },
});

export const clearChatMemory = tool({
  description: `Clear all stored memory for a specific chat. Use when the user wants to reset what the bot knows about someone.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat username or numeric ID"),
  }),
  execute: async ({ chat_id }) => {
    const params = new URLSearchParams({ chat_id });
    const data = await telegramFetch<any>(`/nlp/memory?${params.toString()}`, {
      method: "DELETE",
    });
    return data;
  },
});

// ─── Conversation Analytics ───────────────────────────────────

export const getConversationAnalytics = tool({
  description: `Get detailed conversation analytics for a chat. Returns: response time analysis (how fast you both reply), message ratio, peak activity hours, sentiment trends over time, emoji usage stats, average message lengths, and who initiates conversations more. Use this to understand conversation dynamics and patterns.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat username (e.g., '@username') or numeric chat ID"),
    limit: z.number().min(50).max(500).default(200).describe("Number of messages to analyze"),
  }),
  execute: async ({ chat_id, limit }) => {
    const params = new URLSearchParams({ limit: String(limit) });
    const data = await telegramFetch<any>(`/analytics/${chat_id}?${params.toString()}`);
    return data;
  },
});

// ─── Message Quality Scoring ──────────────────────────────────

export const scoreMessage = tool({
  description: `Score a proposed message before sending it. Checks: length appropriateness (compared to their messages), formality level, language match (English/Russian), emoji usage, AI-sounding phrases, sentiment match, and naturalness. Returns a score (0-100), letter grade (A-F), and specific feedback. Use this to quality-check messages before sending.`,
  inputSchema: z.object({
    message: z.string().describe("The proposed message to score"),
    chat_id: z.string().describe("The chat this message would be sent to (for context comparison)"),
  }),
  execute: async ({ message, chat_id }) => {
    const data = await telegramFetch<any>(`/message/score`, {
      method: "POST",
      body: JSON.stringify({ message, chat_id }),
    });
    return data;
  },
});

// ─── Advanced Intelligence Tools ──────────────────────────────

export const analyzeChatV2 = tool({
  description: `Run the ENHANCED V2 NLP analysis on a chat. Returns everything from V1 PLUS: time-of-day awareness, passive-aggression detection, sarcasm detection, testing behavior detection (when she's testing you), urgency detection, relationship health score, conflict resolution guidance with de-escalation strategies, smart reply delay recommendations, response staleness check, Russian cultural context (endearments, abbreviations, diminutives), and proactive engagement suggestions. This is the most comprehensive analysis tool available.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat username (e.g., '@username') or numeric chat ID"),
  }),
  execute: async ({ chat_id }) => {
    const params = new URLSearchParams({ chat_id });
    const data = await telegramFetch<any>(`/nlp/analyze-v2?${params.toString()}`);
    return data;
  },
});

export const getRelationshipHealth = tool({
  description: `Get a comprehensive conversation health score (0-100, grade A-F) for a chat. Analyzes: positive-to-negative sentiment ratio (Gottman's 5:1 rule), message balance, engagement trends, question ratio, emoji warmth, intimacy level, and recent conflict impact. Returns a detailed breakdown of each signal with individual scores.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat username or numeric ID"),
    limit: z.number().min(50).max(500).default(100).describe("Messages to analyze"),
  }),
  execute: async ({ chat_id, limit }) => {
    const params = new URLSearchParams({ chat_id, limit: String(limit) });
    const data = await telegramFetch<any>(`/relationship/health?${params.toString()}`);
    return data;
  },
});

export const getProactiveSuggestions = tool({
  description: `Get AI-generated proactive message suggestions based on time of day, conversation memory, and context. Returns: contextually appropriate messages, topic suggestions, check-in messages — all personalized to the conversation and timed appropriately. Use this when the user asks "what should I text?" or "should I message them?".`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat username or numeric ID"),
  }),
  execute: async ({ chat_id }) => {
    const params = new URLSearchParams({ chat_id });
    const data = await telegramFetch<any>(`/proactive/suggestions?${params.toString()}`);
    return data;
  },
});

export const checkMessageStaleness = tool({
  description: `Check if a proposed message is too similar to something you've recently sent in this chat. Prevents repetitive messaging by comparing against the last 50 responses using word similarity. Returns: is_stale (boolean), similarity score (0-1), and the similar past message if found.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat username or numeric ID"),
    message: z.string().describe("The proposed message to check"),
  }),
  execute: async ({ chat_id, message }) => {
    const data = await telegramFetch<any>(`/message/staleness`, {
      method: "POST",
      body: JSON.stringify({ chat_id, message }),
    });
    return data;
  },
});

// ─── V3 Deep Learning Intelligence Tools ─────────────────────

export const analyzeChatV3 = tool({
  description: `Run the most advanced V3 NLP analysis powered by deep learning. Combines ALL V2 features PLUS: transformer-based sentiment (DistilBERT), multi-label emotion detection (7 emotions with confidence: anger/disgust/fear/joy/neutral/sadness/surprise), zero-shot intent classification (14 intents), zero-shot topic classification (14 topics), custom neural network predictions (CNN/Attention), semantic similarity analysis, conversation dynamics modeling (momentum/reciprocity/topic coherence/emotional trajectory), confidence-weighted ensemble of all signals, and relevant memory retrieval using sentence embeddings. This is the ULTIMATE analysis tool - use this instead of V2 when you need the deepest understanding.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat username (e.g., '@username') or numeric chat ID"),
  }),
  execute: async ({ chat_id }) => {
    const params = new URLSearchParams({ chat_id });
    try {
      const data = await telegramFetch<any>(`/nlp/analyze-v3?${params.toString()}`);
      return data;
    } catch {
      // Fall back to V2 if V3 endpoint not available
      const data = await telegramFetch<any>(`/nlp/analyze-v2?${params.toString()}`);
      return { ...data, _fallback: "v2" };
    }
  },
});

export const scoreMessageV3 = tool({
  description: `Score a proposed message using neural quality analysis (V3). Much more sophisticated than V1 scoring. Uses: semantic relevance scoring (embedding similarity to their message), emotional tone matching (transformer emotion detection), AI-phrase detection, formality scoring, semantic staleness check (embedding-based deduplication), and length appropriateness. Returns score (0-100), grade (A-F), detailed dimension scores, and specific actionable feedback.`,
  inputSchema: z.object({
    message: z.string().describe("The proposed message to score"),
    chat_id: z.string().describe("The chat this message would be sent to"),
  }),
  execute: async ({ message, chat_id }) => {
    try {
      const data = await telegramFetch<any>(`/message/score-v3`, {
        method: "POST",
        body: JSON.stringify({ message, chat_id }),
      });
      return data;
    } catch {
      // Fall back to V1 scoring
      const data = await telegramFetch<any>(`/message/score`, {
        method: "POST",
        body: JSON.stringify({ message, chat_id }),
      });
      return { ...data, _fallback: "v1" };
    }
  },
});

export const checkStalenessV3 = tool({
  description: `Check message staleness using semantic similarity (V3). Instead of word overlap, uses sentence-transformer embeddings to detect semantically similar past responses even if worded differently. Much more accurate at preventing repetitive messaging.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat username or numeric ID"),
    message: z.string().describe("The proposed message to check"),
  }),
  execute: async ({ chat_id, message }) => {
    try {
      const data = await telegramFetch<any>(`/message/staleness-v3`, {
        method: "POST",
        body: JSON.stringify({ chat_id, message }),
      });
      return data;
    } catch {
      const data = await telegramFetch<any>(`/message/staleness`, {
        method: "POST",
        body: JSON.stringify({ chat_id, message }),
      });
      return { ...data, _fallback: "v1" };
    }
  },
});

export const getDLStatus = tool({
  description: `Get deep learning system status. Shows: which transformer models are loaded, available custom classifiers, device (CPU/GPU/MPS), loaded pipelines, and system health. Use this to check if the neural analysis features are available and working.`,
  inputSchema: z.object({}),
  execute: async () => {
    try {
      const data = await telegramFetch<any>(`/dl/status`);
      return data;
    } catch {
      return { status: "unavailable", message: "DL system not running" };
    }
  },
});

export const trainModels = tool({
  description: `Trigger training of custom NLP models. Trains on 1200+ labeled examples covering: romantic intent (12 categories), conversation stage (7 stages), emotional tone (11 tones). Uses sentence-transformer embeddings + multiple classifiers (LogisticRegression, RandomForest, GradientBoosting, SVM) and picks the best. Set include_neural=true to also train CNN/Attention neural networks (slower but more sophisticated).`,
  inputSchema: z.object({
    task: z.enum(["all", "romantic_intent", "conversation_stage", "emotional_tone"]).default("all").describe("Which classifier to train"),
    include_neural: z.boolean().default(false).describe("Also train CNN/Attention networks (slower)"),
  }),
  execute: async ({ task, include_neural }) => {
    const params = new URLSearchParams({ task, include_neural: String(include_neural) });
    const data = await telegramFetch<any>(`/dl/train?${params.toString()}`, {
      method: "POST",
    });
    return data;
  },
});

// ============= V4 SOPHISTICATED ENGINE TOOLS =============

export const analyzeV4 = tool({
  description: `Run the most advanced V4 analysis on a chat. Combines ALL intelligence engines: Conversation Intelligence (state machine, weighted context, few-shot examples, goal tracking), Emotional Intelligence (VAD profiling, validation guidance, attachment style detection, temporal tracking), Style Engine (communication fingerprint, mirroring directives, personality modeling), Three-Tier Memory (semantic facts, episodic events, procedural patterns), and Reasoning Engine (chain-of-thought, multi-hypothesis, model scaling). This is the ULTIMATE analysis tool.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username to analyze"),
  }),
  execute: async ({ chat_id }) => {
    try {
      const data = await telegramFetch<any>(`/engine/analyze-v4?chat_id=${encodeURIComponent(chat_id)}`);
      return data;
    } catch (e) {
      // Fallback to V3
      try {
        const v3 = await telegramFetch<any>(`/nlp/analyze-v3?chat_id=${encodeURIComponent(chat_id)}`);
        return { ...v3, _fallback: "v3" };
      } catch {
        const v2 = await telegramFetch<any>(`/nlp/analyze-v2?chat_id=${encodeURIComponent(chat_id)}`);
        return { ...v2, _fallback: "v2" };
      }
    }
  },
});

export const getEngineStatus = tool({
  description: `Get the status of all V4 sophistication engines. Shows which engines are loaded (Conversation, Emotional Intelligence, Style, Memory, Reasoning) and their available functions. Use this to check system health.`,
  inputSchema: z.object({}),
  execute: async () => {
    try {
      const data = await telegramFetch<any>(`/engine/status`);
      return data;
    } catch {
      return { error: "Engine status unavailable" };
    }
  },
});

export const getEmotionalHistory = tool({
  description: `Get the emotional history timeline for a chat partner. Shows their emotional patterns over time, baseline mood, temporal patterns (morning vs evening moods, day-of-week patterns), emotional streaks, and whether you should check in on them based on recent emotional state.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
  }),
  execute: async ({ chat_id }) => {
    try {
      const data = await telegramFetch<any>(`/engine/emotional-history?chat_id=${encodeURIComponent(chat_id)}`);
      return data;
    } catch {
      return { error: "Emotional history unavailable" };
    }
  },
});

export const getStyleProfile = tool({
  description: `Get the communication style profile and personality config for a chat partner. Shows their texting patterns (message length, emoji usage, formality, humor frequency, affection level), the bot's personality configuration, and style mirroring recommendations.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
  }),
  execute: async ({ chat_id }) => {
    try {
      const data = await telegramFetch<any>(`/engine/style-profile?chat_id=${encodeURIComponent(chat_id)}`);
      return data;
    } catch {
      return { error: "Style profile unavailable" };
    }
  },
});

export const getAdvancedMemory = tool({
  description: `Get the three-tier memory system for a chat. Returns: Semantic Memory (facts: name, age, location, occupation, preferences, family, pets, important dates), Episodic Memory (conversation events, milestones like first "I love you", shared references/inside jokes), and Procedural Memory (learned communication patterns, what worked/didn't work in past interactions).`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
  }),
  execute: async ({ chat_id }) => {
    try {
      const data = await telegramFetch<any>(`/engine/memory?chat_id=${encodeURIComponent(chat_id)}`);
      return data;
    } catch {
      return { error: "Advanced memory unavailable" };
    }
  },
});

export const consolidateMemory = tool({
  description: `Run memory consolidation for a chat. This compresses old episodic memories, deduplicates semantic facts, and organizes the knowledge graph. Run this periodically to keep memory efficient and well-organized.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
  }),
  execute: async ({ chat_id }) => {
    try {
      const params = new URLSearchParams({ chat_id });
      const data = await telegramFetch<any>(`/engine/consolidate-memory?${params.toString()}`, {
        method: "POST",
      });
      return data;
    } catch {
      return { error: "Memory consolidation unavailable" };
    }
  },
});

// ============= V5 ENHANCED PSYCHOLOGICAL TOOLS =============

export const analyzeV5 = tool({
  description: `Run V5 ULTIMATE analysis combining ALL engines + psychological datasets.
Includes everything from V4 plus: Gottman's Four Horsemen & 5:1 ratio, Knapp's relationship stages,
Plutchik's Wheel of Emotions (32 emotions), GoEmotions (27 categories), Love Languages detection,
Big Five personality traits, CBT cognitive distortion detection, NVC quality analysis,
Thomas-Kilmann conflict modes, Chain of Empathy reasoning, behavioral pattern detection
(ghosting, breadcrumbing, love bombing), relationship trajectory tracking.
This is the MOST comprehensive analysis available.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
  }),
  execute: async ({ chat_id }) => {
    try {
      const params = new URLSearchParams({ chat_id });
      const data = await telegramFetch<any>(`/engine/analyze-v5?${params.toString()}`);
      return data;
    } catch {
      try {
        const params = new URLSearchParams({ chat_id });
        return await telegramFetch<any>(`/engine/analyze-v4?${params.toString()}`);
      } catch {
        return { error: "V5 analysis unavailable" };
      }
    }
  },
});

export const getPsychologicalAnalysis = tool({
  description: `Run comprehensive psychological analysis on a conversation.
Detects: Gottman's Four Horsemen (criticism, contempt, defensiveness, stonewalling),
emotional bids for connection, repair attempts, positive:negative ratio,
Love Languages, Knapp's relationship stage, cognitive distortions (13 types from CBT),
conflict modes (Thomas-Kilmann), NVC quality, and Big Five personality traits.
Returns warnings and overall health assessment.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
  }),
  execute: async ({ chat_id }) => {
    try {
      const params = new URLSearchParams({ chat_id });
      return await telegramFetch<any>(`/engine/psychological-analysis?${params.toString()}`);
    } catch {
      return { error: "Psychological analysis unavailable" };
    }
  },
});

export const getGottmanRatio = tool({
  description: `Get Gottman's positive-to-negative interaction ratio for a chat.
Research shows stable relationships maintain at least 5:1 ratio (5 positive for every 1 negative).
Below 1:1 is critical. Returns ratio, assessment, and whether it's healthy.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
  }),
  execute: async ({ chat_id }) => {
    try {
      const params = new URLSearchParams({ chat_id });
      return await telegramFetch<any>(`/engine/gottman-ratio?${params.toString()}`);
    } catch {
      return { error: "Gottman ratio unavailable" };
    }
  },
});

export const getLoveLanguage = tool({
  description: `Detect a contact's primary communication preference using Chapman's Five Love Languages framework.
Categories: Words of Affirmation, Quality Time, Acts of Service, Receiving Gifts, Physical Touch.
Helps tailor messages to what resonates most with the recipient.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
  }),
  execute: async ({ chat_id }) => {
    try {
      const params = new URLSearchParams({ chat_id });
      return await telegramFetch<any>(`/engine/love-language?${params.toString()}`);
    } catch {
      return { error: "Love language detection unavailable" };
    }
  },
});

export const getRelationshipStage = tool({
  description: `Detect the current relationship development stage using Knapp's Relational Model.
10 stages: Coming Together (initiating, experimenting, intensifying, integrating, bonding)
and Coming Apart (differentiating, circumscribing, stagnating, avoiding, terminating).
Includes warning level if in a "coming apart" phase.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
  }),
  execute: async ({ chat_id }) => {
    try {
      const params = new URLSearchParams({ chat_id });
      return await telegramFetch<any>(`/engine/relationship-stage?${params.toString()}`);
    } catch {
      return { error: "Relationship stage detection unavailable" };
    }
  },
});

export const getRelationshipTrajectory = tool({
  description: `Get relationship trajectory over time. Tracks:
sentiment trend (improving/stable/declining), Gottman ratio history,
Knapp stage transitions, and overall trajectory health.
Shows how the relationship is evolving over time, not just current state.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
  }),
  execute: async ({ chat_id }) => {
    try {
      const params = new URLSearchParams({ chat_id });
      return await telegramFetch<any>(`/engine/relationship-trajectory?${params.toString()}`);
    } catch {
      return { error: "Relationship trajectory unavailable" };
    }
  },
});

export const getBigFive = tool({
  description: `Detect Big Five (OCEAN) personality traits from messaging patterns.
Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism.
Based on Pennebaker's LIWC research linking linguistic markers to personality traits.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
  }),
  execute: async ({ chat_id }) => {
    try {
      const params = new URLSearchParams({ chat_id });
      return await telegramFetch<any>(`/engine/big-five?${params.toString()}`);
    } catch {
      return { error: "Big Five analysis unavailable" };
    }
  },
});

export const getBehavioralPatterns = tool({
  description: `Detect concerning behavioral patterns in a conversation:
ghosting (sudden communication cessation), breadcrumbing (sporadic minimal messages),
love bombing (excessive early intensity), stonewalling (repeated withdrawal),
hot-cold cycling (alternating engagement/withdrawal), engagement decline.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
  }),
  execute: async ({ chat_id }) => {
    try {
      const params = new URLSearchParams({ chat_id });
      return await telegramFetch<any>(`/engine/behavioral-patterns?${params.toString()}`);
    } catch {
      return { error: "Behavioral pattern detection unavailable" };
    }
  },
});

// ─── System Dashboard & Model Tools ──────────────────────────

export const getDashboard = tool({
  description: `Get aggregated system dashboard: auto-reply status, engine list with function counts, trained model accuracies, feature flags, media AI status, and recent activity. Single call for full system overview.`,
  inputSchema: z.object({}),
  execute: async () => {
    try {
      return await telegramFetch<any>(`/dashboard`);
    } catch {
      return { error: "Dashboard unavailable" };
    }
  },
});

export const getFeatures = tool({
  description: `Get all auto-reply feature flags with their current ON/OFF status. Shows: late_night_mode, strategic_silence, quote_reply, smart_reactions, message_editing, gif_sticker_reply, typing_awareness, online_status_aware, proactive_enabled, proactive_morning, proactive_night.`,
  inputSchema: z.object({}),
  execute: async () => {
    try {
      return await telegramFetch<any>(`/auto-reply/features`);
    } catch {
      return { error: "Features unavailable" };
    }
  },
});

export const toggleFeature = tool({
  description: `Toggle an auto-reply feature ON or OFF. Provide the feature name and the desired value.`,
  inputSchema: z.object({
    feature: z.string().describe("Feature name (e.g., 'strategic_silence', 'late_night_mode')"),
    enabled: z.boolean().describe("true to enable, false to disable"),
  }),
  execute: async ({ feature, enabled }) => {
    try {
      return await telegramFetch<any>(`/auto-reply/features`, {
        method: "PUT",
        body: JSON.stringify({ [feature]: enabled }),
      });
    } catch {
      return { error: "Feature toggle failed" };
    }
  },
});

export const getModelsStatus = tool({
  description: `Get detailed ML model status: sklearn classifiers (romantic_intent, conversation_stage, emotional_tone) with accuracy, classifier type, class count, training size; and neural networks (TextCNN, EmotionAttn) with accuracy and class count.`,
  inputSchema: z.object({}),
  execute: async () => {
    try {
      return await telegramFetch<any>(`/models/status`);
    } catch {
      return { error: "Model status unavailable" };
    }
  },
});

// ═══════════════════════════════════════════════════════════════
// UNIFIED CONTEXT INTELLIGENCE — single endpoint for EVERYTHING
// ═══════════════════════════════════════════════════════════════

export const getContextIntelligence = tool({
  description: `Get the FULL unified context intelligence for a chat — the single most comprehensive analysis endpoint. Combines ALL engines into one response:
- Context Intelligence: active threads, unanswered questions, conversation arc, their intent
- NLP Analysis: sentiment, language, topics, conversation stage, passive-aggression, sarcasm, urgency
- Emotional State: primary emotion, intensity, attachment style, validation guidance
- Personality: Big Five, communication preferences, archetype
- Predictions: engagement score, ghost risk, conflict risk
- Thinking Engine: situation analysis, Monte Carlo strategy
- Memory: learned facts about the person
- Summary: one-line overview of everything

Use this BEFORE composing a message to understand the full context. This is the brain of the system.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
  }),
  execute: async ({ chat_id }) => {
    try {
      return await telegramFetch<any>(`/advanced/context-intelligence/${chat_id}`);
    } catch (e: any) {
      return { error: e.message };
    }
  },
});

// Export all telegram tools
// ═══════════════════════════════════════════════════════════════
// VOICE CLONING & SYNTHESIS
// ═══════════════════════════════════════════════════════════════

export const getVoiceStatus = tool({
  description: `Get voice engine status — shows available backends (Chatterbox voice cloning, Bark, Edge TTS), whether the user's voice is registered for cloning, supported languages, and cache stats.`,
  inputSchema: z.object({}),
  execute: async () => {
    return await telegramFetch<any>("/voice/status");
  },
});

export const registerVoice = tool({
  description: `Register a voice for cloning. Either provide a local audio file path, or a chat_id to automatically grab the latest voice message from that chat. The voice reference should be 5-10 seconds of clear speech. Once registered, all voice messages will sound like that person. Supports both English and Russian voices.`,
  inputSchema: z.object({
    audio_url: z.string().optional().describe("Path to a local audio file (WAV/OGG/MP3)"),
    chat_id: z.string().optional().describe("Chat ID to grab latest voice message from (e.g. @username or Saved Messages)"),
  }),
  execute: async ({ audio_url, chat_id }) => {
    const params = new URLSearchParams();
    if (audio_url) params.set("audio_url", audio_url);
    if (chat_id) params.set("chat_id", chat_id);
    return await telegramFetch<any>(`/voice/register?${params.toString()}`, { method: "POST" });
  },
});

export const sendVoiceClone = tool({
  description: `Generate a voice message using the cloned voice and send it to a chat. The voice will sound like the registered person, not like an AI. Supports English and Russian text. Use emotion parameter to control expressiveness (neutral, joy, love, anger, sadness, excitement, playful, sarcastic, tired).`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username to send to"),
    text: z.string().describe("Text to speak in the cloned voice"),
    emotion: z.string().optional().describe("Emotion for voice prosody: neutral, joy, love, anger, etc."),
    reply_to: z.number().optional().describe("Message ID to reply to"),
  }),
  execute: async ({ chat_id, text, emotion, reply_to }) => {
    const params = new URLSearchParams({ text });
    if (emotion) params.set("emotion", emotion);
    if (reply_to) params.set("reply_to", String(reply_to));
    return await telegramFetch<any>(`/voice/clone-and-send/${chat_id}?${params.toString()}`, { method: "POST" });
  },
});

export const generateVoice = tool({
  description: `Generate voice audio from text without sending it. Returns the audio file path. Useful for previewing before sending, or for creating audio files. Supports English and Russian.`,
  inputSchema: z.object({
    text: z.string().describe("Text to convert to speech"),
    language: z.string().optional().describe("Language: en, ru, auto"),
    emotion: z.string().optional().describe("Emotion for prosody control"),
    backend: z.string().optional().describe("Backend: auto, chatterbox, bark, edge"),
  }),
  execute: async ({ text, language, emotion, backend }) => {
    const params = new URLSearchParams({ text });
    if (language) params.set("language", language);
    if (emotion) params.set("emotion", emotion);
    if (backend) params.set("backend", backend);
    return await telegramFetch<any>(`/voice/generate?${params.toString()}`, { method: "POST" });
  },
});

// ═══════════════════════════════════════════════════════════════
// INTERVENTION & CONTROL
// ═══════════════════════════════════════════════════════════════

export const intervene = tool({
  description: `Tell the auto-reply bot EXACTLY what to do in a specific chat. One-shot (duration=0) applies to the very next reply only. Persistent (duration>0) applies for N minutes. Examples: "be more aggressive", "apologize and be sweet", "ignore her for now", "switch to Russian", "send a voice note next time".`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
    instruction: z.string().describe("What the bot should do"),
    duration_minutes: z.number().optional().describe("How long (0=one-shot, default)"),
  }),
  execute: async ({ chat_id, instruction, duration_minutes }) => {
    const params = new URLSearchParams({
      instruction,
      duration_minutes: String(duration_minutes || 0),
    });
    return await telegramFetch<any>(`/intervene/${chat_id}?${params.toString()}`, { method: "POST" });
  },
});

export const pauseChat = tool({
  description: `Pause auto-reply for a specific chat so the user can take manual control. The bot will NOT reply in this chat until resumed or the timer expires.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
    minutes: z.number().optional().describe("Pause duration in minutes (default 30)"),
  }),
  execute: async ({ chat_id, minutes }) => {
    const params = new URLSearchParams({ minutes: String(minutes || 30) });
    return await telegramFetch<any>(`/intervene/${chat_id}/pause?${params.toString()}`, { method: "POST" });
  },
});

export const resumeChat = tool({
  description: `Resume auto-reply for a paused chat. Clears all interventions and overrides.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
  }),
  execute: async ({ chat_id }) => {
    return await telegramFetch<any>(`/intervene/${chat_id}/resume`, { method: "POST" });
  },
});

export const getInterventionStatus = tool({
  description: `Get all active interventions, paused chats, and pending overrides.`,
  inputSchema: z.object({}),
  execute: async () => {
    return await telegramFetch<any>("/intervene/status");
  },
});

export const queueMessage = tool({
  description: `Queue a message to be sent in a chat with optional delay. Can send as voice note. Use for scheduled follow-ups, delayed responses, or queuing multiple messages.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
    message: z.string().describe("Message to send"),
    delay_seconds: z.number().optional().describe("Delay before sending (default 0)"),
    as_voice: z.boolean().optional().describe("Send as voice note (default false)"),
  }),
  execute: async ({ chat_id, message, delay_seconds, as_voice }) => {
    const params = new URLSearchParams({ message });
    if (delay_seconds) params.set("delay_seconds", String(delay_seconds));
    if (as_voice) params.set("as_voice", "true");
    return await telegramFetch<any>(`/intervene/${chat_id}/queue?${params.toString()}`, { method: "POST" });
  },
});

export const listVoices = tool({
  description: `List all available voice references — your voice, named personas, and per-chat voice assignments.`,
  inputSchema: z.object({}),
  execute: async () => {
    return await telegramFetch<any>("/voice/voices");
  },
});

export const assignVoice = tool({
  description: `Assign a specific voice to a chat. Each chat can have its own voice for voice messages.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Chat ID or @username"),
    voice_path: z.string().describe("Path to the voice file to assign"),
  }),
  execute: async ({ chat_id, voice_path }) => {
    const params = new URLSearchParams({ voice_path });
    return await telegramFetch<any>(`/voice/assign/${chat_id}?${params.toString()}`, { method: "POST" });
  },
});

// ── Voice Calls (Private + Group) ──────────────────────────

export const getCallStatus = tool({
  description: `Get call engine status — shows active calls, bridge health, and capabilities for both private and group calls.`,
  inputSchema: z.object({}),
  execute: async () => {
    return await telegramFetch<any>("/call/status");
  },
});

export const startCallBridge = tool({
  description: `Start the call bridge subprocess (Python 3.10 + tgcalls). Must be running before making/receiving calls.`,
  inputSchema: z.object({}),
  execute: async () => {
    return await telegramFetch<any>("/call/start-bridge", { method: "POST" });
  },
});

export const makeCall = tool({
  description: `Make a private voice call to a user. Optionally include an initial message to say when they pick up.`,
  inputSchema: z.object({
    user_id: z.string().describe("User ID or @username to call"),
    initial_message: z.string().optional().describe("First thing to say when connected"),
  }),
  execute: async ({ user_id, initial_message }) => {
    const params = new URLSearchParams();
    if (initial_message) params.set("initial_message", initial_message);
    const qs = params.toString();
    return await telegramFetch<any>(`/call/make/${user_id}${qs ? `?${qs}` : ""}`, { method: "POST" });
  },
});

export const acceptCall = tool({
  description: `Accept an incoming voice call.`,
  inputSchema: z.object({
    user_id: z.string().describe("User ID of the caller"),
  }),
  execute: async ({ user_id }) => {
    return await telegramFetch<any>(`/call/accept/${user_id}`, { method: "POST" });
  },
});

export const declineCall = tool({
  description: `Decline an incoming voice call.`,
  inputSchema: z.object({
    user_id: z.string().describe("User ID of the caller"),
  }),
  execute: async ({ user_id }) => {
    return await telegramFetch<any>(`/call/decline/${user_id}`, { method: "POST" });
  },
});

export const hangupCall = tool({
  description: `Hang up an active voice call (private or group).`,
  inputSchema: z.object({
    user_id: z.string().describe("User/Chat ID to hang up"),
  }),
  execute: async ({ user_id }) => {
    return await telegramFetch<any>(`/call/hangup/${user_id}`, { method: "POST" });
  },
});

export const speakInCall = tool({
  description: `Speak text in an active call using TTS voice cloning. Works for both private and group calls.`,
  inputSchema: z.object({
    user_id: z.string().describe("User/Chat ID of the active call"),
    text: z.string().describe("Text to speak"),
    emotion: z.string().optional().describe("Emotion: neutral, happy, sad, flirty, angry"),
    language: z.string().optional().describe("Language code or 'auto'"),
  }),
  execute: async ({ user_id, text, emotion, language }) => {
    const params = new URLSearchParams({ text });
    if (emotion) params.set("emotion", emotion);
    if (language) params.set("language", language);
    return await telegramFetch<any>(`/call/speak/${user_id}?${params.toString()}`, { method: "POST" });
  },
});

export const listenInCall = tool({
  description: `Get real-time transcription of what the other party is saying in an active call.`,
  inputSchema: z.object({
    user_id: z.string().describe("User/Chat ID of the active call"),
  }),
  execute: async ({ user_id }) => {
    return await telegramFetch<any>(`/call/listen/${user_id}`);
  },
});

export const joinGroupCall = tool({
  description: `Join a group voice chat (voice call in a group or channel).`,
  inputSchema: z.object({
    chat_id: z.string().describe("Group/Channel chat ID"),
    initial_message: z.string().optional().describe("First thing to say when joined"),
  }),
  execute: async ({ chat_id, initial_message }) => {
    const params = new URLSearchParams();
    if (initial_message) params.set("initial_message", initial_message);
    const qs = params.toString();
    return await telegramFetch<any>(`/call/group/join/${chat_id}${qs ? `?${qs}` : ""}`, { method: "POST" });
  },
});

export const leaveGroupCall = tool({
  description: `Leave a group voice chat.`,
  inputSchema: z.object({
    chat_id: z.string().describe("Group/Channel chat ID"),
  }),
  execute: async ({ chat_id }) => {
    return await telegramFetch<any>(`/call/group/leave/${chat_id}`, { method: "POST" });
  },
});

export const setCallAutonomy = tool({
  description: `Enable or disable autonomous mode for an active call. When enabled, the bot listens to the other person, transcribes their speech with Whisper, generates AI responses using Claude, and speaks them back using TTS/voice cloning — full conversational autonomy without human intervention.`,
  inputSchema: z.object({
    chat_id: z.string().describe("User/Chat ID of the active call"),
    enabled: z.boolean().describe("true to enable autonomy, false to disable"),
    language: z.string().optional().describe("Language code (e.g. 'ru', 'en') or 'auto'"),
  }),
  execute: async ({ chat_id, enabled, language }) => {
    const params = new URLSearchParams({
      enabled: String(enabled),
    });
    if (language) params.set("language", language);
    return await telegramFetch<any>(`/call/autonomy/${chat_id}?${params.toString()}`, { method: "POST" });
  },
});

export const setAutoAcceptCalls = tool({
  description: `Configure auto-accept for incoming calls. When enabled, incoming calls are automatically accepted. With autonomy=true, the bot also starts speaking autonomously — fully hands-free call handling.`,
  inputSchema: z.object({
    enabled: z.boolean().describe("true to auto-accept, false to disable"),
    with_autonomy: z.boolean().optional().describe("Also enable autonomy when accepting"),
  }),
  execute: async ({ enabled, with_autonomy }) => {
    const params = new URLSearchParams({
      enabled: String(enabled),
    });
    if (with_autonomy !== undefined) params.set("with_autonomy", String(with_autonomy));
    return await telegramFetch<any>(`/call/auto-accept?${params.toString()}`, { method: "POST" });
  },
});

export const telegramTools = {
  // Core
  getChats,
  getMessages,
  sendMessage,
  scheduleMessage,
  getChat,
  searchContacts,
  // History & Search
  getHistory,
  searchMessages,
  // Reactions & Replies
  sendReaction,
  replyToMessage,
  // Edit & Delete
  editMessage,
  deleteMessage,
  // Forward & Pin
  forwardMessage,
  pinMessage,
  markAsRead,
  // User Info
  getUserStatus,
  getUserPhotos,
  // Media
  searchGifs,
  // Auto-Reply Management
  setAutoReplyInstructions,
  getAutoReplyStatus,
  toggleAutoReply,
  getAutoReplyLog,
  // NLP & Conversation Intelligence
  analyzeChat,
  getChatMemory,
  listChatMemories,
  addMemoryNote,
  clearChatMemory,
  // Analytics & Quality
  getConversationAnalytics,
  scoreMessage,
  // Advanced Intelligence (V2)
  analyzeChatV2,
  getRelationshipHealth,
  getProactiveSuggestions,
  checkMessageStaleness,
  // Deep Learning Intelligence (V3)
  analyzeChatV3,
  scoreMessageV3,
  checkStalenessV3,
  getDLStatus,
  trainModels,
  // V4 Sophisticated Engine Intelligence
  analyzeV4,
  getEngineStatus,
  getEmotionalHistory,
  getStyleProfile,
  getAdvancedMemory,
  consolidateMemory,
  // V5 Enhanced Psychological Intelligence
  analyzeV5,
  getPsychologicalAnalysis,
  getGottmanRatio,
  getLoveLanguage,
  getRelationshipStage,
  getRelationshipTrajectory,
  getBigFive,
  getBehavioralPatterns,
  // System Dashboard & Models
  getDashboard,
  getFeatures,
  toggleFeature,
  getModelsStatus,
  // Unified Context Intelligence
  getContextIntelligence,
  // Voice Cloning & Synthesis
  getVoiceStatus,
  registerVoice,
  sendVoiceClone,
  generateVoice,
  listVoices,
  assignVoice,
  // Intervention & Control
  intervene,
  pauseChat,
  resumeChat,
  getInterventionStatus,
  queueMessage,
  // Voice Calls (Private + Group)
  getCallStatus,
  startCallBridge,
  makeCall,
  acceptCall,
  declineCall,
  hangupCall,
  speakInCall,
  listenInCall,
  joinGroupCall,
  leaveGroupCall,
  setCallAutonomy,
  setAutoAcceptCalls,
};
