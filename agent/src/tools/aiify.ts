/**
 * AI-ify tool - Transform messages with intelligent style adaptation
 * Uses the agent's LLM to generate contextually appropriate responses
 */

import { tool } from "ai";
import { z } from "zod";

export const aiifyMessage = tool({
  description: `Transform a received message into a contextually appropriate response. This tool helps craft the ideal reply by:
1. Analyzing the incoming message's tone and context
2. Searching conversation templates for relevant content
3. Generating a response that matches the desired style

Use this when you want to AI-enhance a response to any message.`,
  inputSchema: z.object({
    incoming_message: z.string().describe("The message received that you want to respond to"),
    style: z
      .enum(["engaging", "thoughtful", "funny", "witty", "warm", "playful", "professional"])
      .default("witty")
      .describe("The tone/style you want for the response"),
    context: z
      .string()
      .optional()
      .describe("Additional context about the conversation, shared references, or current situation"),
  }),
  execute: async ({ incoming_message, style, context }) => {
    return {
      instruction: `Generate a ${style} response to: "${incoming_message}"`,
      style,
      originalMessage: incoming_message,
      context: context || "No additional context",
      suggestion: `Use searchConversationTemplates to find relevant content, then craft a ${style} response that:
- Matches the energy of the incoming message
- Shows personality and wit
- Feels natural and not forced
- Could include a callback to something they said`,
    };
  },
});

export const aiifyTools = {
  aiifyMessage,
};
