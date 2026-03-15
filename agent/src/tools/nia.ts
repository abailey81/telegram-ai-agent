/**
 * Conversation templates & Nia tools for the AI agent
 * Local conversation template search + optional Nia API integration
 */

import { tool } from "ai";
import { z } from "zod";
import { config } from "../config";
import { pickupLines, type PickupLine } from "../data/pickup-lines";

const API = config.niaApiBase;

// ─── Local search engine ─────────────────────────────────────

function searchLocal(query: string, language?: string): { text: string; category: string; tone: string; score: number }[] {
  const queryLower = query.toLowerCase();
  const queryWords = queryLower.split(/\s+/).filter(w => w.length > 2);

  const scored = pickupLines.map(line => {
    let score = 0;

    // Language filter boost/penalty
    if (language) {
      if (line.language === language) score += 3;
      else if (line.language && line.language !== language) score -= 5;
    } else {
      // If no language specified, slightly prefer English (no language field)
      if (line.language) score -= 1;
    }

    // Tag matching (strongest signal)
    for (const tag of line.tags) {
      for (const word of queryWords) {
        if (tag.includes(word) || word.includes(tag)) {
          score += 2;
        }
      }
    }

    // Category matching
    for (const word of queryWords) {
      if (line.category.includes(word)) score += 2;
    }

    // Tone matching
    for (const word of queryWords) {
      if (line.tone.includes(word)) score += 2;
    }

    // Text content matching
    for (const word of queryWords) {
      if (line.text.toLowerCase().includes(word)) score += 1;
    }

    // Boost for specific context keywords — category matches
    const categoryBoosts: Record<string, string[]> = {
      "morning": ["morning", "wake up", "good morning"],
      "night": ["night", "goodnight", "sleep", "dream", "good night"],
      "opener": ["opener", "start", "first message", "ice breaker", "conversation starter"],
      "flirty": ["flirty", "flirt", "smooth", "seductive"],
      "funny": ["funny", "joke", "humor", "laugh", "pun"],
      "cheesy": ["cheesy", "corny", "cringe", "over the top"],
      "clever": ["clever", "witty", "smart", "intellectual"],
      "compliment": ["compliment", "beautiful", "pretty", "gorgeous", "eyes", "smile", "voice", "style"],
      "compliment-reply": ["compliment reply", "return compliment"],
      "apology": ["sorry", "apologize", "apology", "forgive", "messed up", "make up"],
      "support": ["bad day", "sad", "upset", "cheer up", "comfort", "stressed", "having a rough"],
      "date": ["date", "dinner", "coffee", "meet up", "hang out", "plans", "cook", "movie", "trip"],
      "conversation": ["conversation", "starter", "topic", "talk about", "question", "deep"],
      "romantic": ["romantic", "love", "deep", "feelings", "heart"],
      "jealousy": ["jealous", "jealousy", "possessive"],
      "jealousy-reply": ["jealous", "reassure", "only you"],
      "missing": ["long distance", "far away", "distance", "miss", "apart", "counting days"],
      "photo-reply": ["photo", "selfie", "pic", "picture", "sent a photo", "sent a pic"],
      "distant": ["cold", "distant", "ignoring", "not responding", "dry", "quiet", "push away"],
      "angry": ["mad", "angry", "fight", "argument", "upset with me"],
      "bored": ["bored", "boring", "nothing to do", "entertain"],
      "what-doing": ["what doing", "what are you doing", "wyd", "whatcha doing"],
      "love-reply": ["do you love me", "love me", "prove it"],
      "future": ["future", "marry", "kids", "move in", "together forever", "5 years"],
      "anniversary": ["anniversary", "birthday", "special", "celebrate", "valentines", "christmas"],
      "pet-name": ["pet name", "nickname", "call me", "babe", "baby", "honey"],
      "excited": ["excited", "happy", "great news", "amazing news", "proud"],
      "work-stress": ["work", "school", "exam", "deadline", "tired", "stress"],
      "voice-selfie-reply": ["voice", "voice message", "selfie", "story", "voice note"],
      "humor": ["meme", "joke", "lol", "haha", "internet", "pov", "red flag"],
      "emoji": ["emoji", "casual text", "cute text"],
      "returning": ["haven't texted", "been away", "ghosted", "disappeared", "MIA"],
      "tease-reply": ["tease", "teasing", "banter", "comeback"],
    };
    for (const [cat, keywords] of Object.entries(categoryBoosts)) {
      if (keywords.some(kw => queryLower.includes(kw)) && line.category === cat) {
        score += 3;
      }
    }

    // Boost for tone keywords
    const toneBoosts: Record<string, string[]> = {
      "flirty": ["flirty", "flirt", "seductive", "bold"],
      "funny": ["funny", "humor", "joke", "laugh"],
      "romantic": ["romantic", "deep", "love", "heartfelt"],
      "cheesy": ["cheesy", "corny", "cringe"],
      "clever": ["clever", "witty", "smart", "intellectual"],
      "sweet": ["sweet", "cute", "adorable", "gentle"],
      "supportive": ["supportive", "caring", "comfort", "empathetic"],
      "playful": ["playful", "teasing", "banter", "tease"],
      "sincere": ["sincere", "honest", "genuine", "real", "vulnerable"],
    };
    for (const [tone, keywords] of Object.entries(toneBoosts)) {
      if (keywords.some(kw => queryLower.includes(kw)) && line.tone === tone) {
        score += 3;
      }
    }

    // Language boosts
    if (queryLower.includes("russian") && line.language === "ru") score += 5;
    if (queryLower.includes("русск") && line.language === "ru") score += 5;
    if (queryLower.includes("по-русски") && line.language === "ru") score += 5;

    // Night context
    if (queryLower.includes("night") && line.tags.includes("late night")) score += 3;
    // Miss context
    if (queryLower.includes("miss") && line.tags.includes("miss")) score += 3;

    return { text: line.text, category: line.category, tone: line.tone, score };
  });

  return scored
    .filter(s => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 8);
}

// ─── Nia API helper (optional) ───────────────────────────────

interface SearchResponse {
  results?: { content: string; source?: string; path?: string; url?: string; score?: number }[];
  answer?: string;
  sources?: any[];
}

async function niaFetch<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  if (!config.niaApiKey) {
    throw new Error("NIA_API_KEY is not configured");
  }

  const url = `${API}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      Authorization: `Bearer ${config.niaApiKey}`,
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Nia API error: ${error}`);
  }

  return response.json();
}

// ─── Tools ───────────────────────────────────────────────────

export const searchConversationTemplates = tool({
  description: `Search the conversation template knowledge base. Use this to find relevant conversation starters, response templates, or contextual responses based on situation. Supports English and Russian.`,
  inputSchema: z.object({
    query: z
      .string()
      .describe(
        "What to search for. Be specific about the context, mood, or topic (e.g., 'funny conversation starter', 'supportive response to bad day', 'witty comeback to teasing', 'russian sweet compliment')"
      ),
    language: z
      .string()
      .optional()
      .describe("Filter by language: 'ru' for Russian, omit for English"),
  }),
  execute: async ({ query, language }) => {
    const results = searchLocal(query, language);

    if (results.length > 0) {
      return {
        results: results.map(r => ({
          content: r.text,
          category: r.category,
          tone: r.tone,
          score: r.score,
        })),
        totalAvailable: pickupLines.length,
      };
    }

    return {
      results: [],
      message: "No templates matched this query. Try broader terms like 'morning', 'funny', 'supportive', 'clever', 'russian'.",
    };
  },
});

export const niaSearch = tool({
  description: `General semantic search via Nia API. Use for broader context when the template search doesn't have what you need.`,
  inputSchema: z.object({
    query: z.string().describe("Search query - ask a natural language question"),
  }),
  execute: async ({ query }) => {
    try {
      const response = await niaFetch<SearchResponse>("/query", {
        method: "POST",
        body: JSON.stringify({
          messages: [{ role: "user", content: query }],
          search_mode: "sources",
          include_sources: true,
        }),
      });

      if (response.sources && response.sources.length > 0) {
        return {
          results: response.sources.slice(0, 5).map((s: any) => ({
            content: s.content || s.text,
            source: s.source_type,
            path: s.path,
            url: s.url,
          })),
          answer: response.answer,
        };
      }

      return { results: [], message: "No results found" };
    } catch (e: any) {
      return { results: [], error: e.message };
    }
  },
});

export const webSearch = tool({
  description: `Search the web for real-time information. Use sparingly - only when you need current information not available in the knowledge base.`,
  inputSchema: z.object({
    query: z.string().describe("Web search query"),
    num_results: z.number().min(1).max(10).default(5).describe("Number of results"),
    category: z
      .enum(["github", "company", "research paper", "news", "tweet", "pdf"])
      .optional()
      .describe("Filter by content category"),
  }),
  execute: async ({ query, num_results, category }) => {
    try {
      const response = await niaFetch<any>("/web-search", {
        method: "POST",
        body: JSON.stringify({
          query,
          num_results,
          ...(category && { category }),
        }),
      });

      return {
        results: response.results?.slice(0, num_results) || [],
      };
    } catch (e: any) {
      return { results: [], error: e.message };
    }
  },
});

// Export all tools
export const niaTools = {
  searchConversationTemplates,
  niaSearch,
  webSearch,
};
