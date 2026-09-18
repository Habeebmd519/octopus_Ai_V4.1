/*
 * Octapus AI V4 - Puter.js User-Pays Agent
 *
 * Include before this file:
 *   <script src="https://js.puter.com/v2/"></script>
 *
 * The browser owns the LLM call. The Flask backend only supplies private
 * application data through tool endpoints. This keeps your 18k+ dataset and
 * API secrets server-side while letting each Puter user pay for their own AI.
 */

(function (global) {
  "use strict";

  const DEFAULTS = {
    apiBase: "",
    model: "gpt-5.6-luna",
    maxToolRounds: 4,
    maxTokens: 1400,
    temperature: 0.2,
    reasoningEffort: "medium",
  };

  function contentOf(response) {
    if (!response) return "";
    if (typeof response === "string") return response;
    if (response.message?.content) {
      if (typeof response.message.content === "string") return response.message.content;
      if (Array.isArray(response.message.content)) {
        return response.message.content.map(x => x?.text || "").join("");
      }
    }
    return response.text || response.content || "";
  }

  async function jsonFetch(url, options) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 30000);
    let res;
    try {
      res = await fetch(url, {

      headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
      ...options,
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); } catch (_) { data = { raw: text }; }
    if (!res.ok) throw new Error(data.message || data.error || `HTTP ${res.status}`);
    return data;
  }

  async function ensurePuterAuth() {
    if (!global.puter) throw new Error("Puter.js is not loaded.");
    if (global.puter.auth?.isSignedIn && !global.puter.auth.isSignedIn()) {
      await global.puter.auth.signIn();
    }
  }

  function compactInitialContext(ctx) {
    return {
      masterIntent: ctx.masterIntent,
      intent: ctx.intent,
      userLocation: ctx.userLocation,
      currentPlace: ctx.currentPlace,
      initialResults: ctx.initialResults,
      live: ctx.live,
      placeCount: ctx.placeCount,
      imageCount: ctx.imageCount,
    };
  }

  function makeTools(apiBase) {
    return [
      {
        type: "function",
        function: {
          name: "search_places",
          description: "Search the private KeralaTour database of 18,000+ places.",
          parameters: {
            type: "object",
            properties: {
              query: { type: "string" },
              limit: { type: "integer", minimum: 1, maximum: 12 }
            },
            required: ["query"]
          },
          strict: true
        }
      },
      {
        type: "function",
        function: {
          name: "get_place",
          description: "Get one specific KeralaTour place.",
          parameters: {
            type: "object",
            properties: {
              place_id: { type: "string" },
              name: { type: "string" }
            }
          },
          strict: true
        }
      },
      {
        type: "function",
        function: {
          name: "search_services",
          description: "Find practical nearby services from OpenStreetMap.",
          parameters: {
            type: "object",
            properties: {
              query: { type: "string" },
              category: { type: "string", enum: ["food", "osm_stay", "osm_health", "osm_transport", "osm_emergency", "osm_money", "osm_fuel"] },
              limit: { type: "integer", minimum: 1, maximum: 12 },
              lat: { type: "number" },
              lng: { type: "number" }
            },
            required: ["query"]
          },
          strict: true
        }
      },
      {
        type: "function",
        function: {
          name: "live_search",
          description: "Search current web information using Tavily.",
          parameters: {
            type: "object",
            properties: {
              query: { type: "string" },
              limit: { type: "integer", minimum: 1, maximum: 5 }
            },
            required: ["query"]
          },
          strict: true
        }
      },
      {
        type: "function",
        function: {
          name: "travel_info",
          description: "Get route distance and duration.",
          parameters: {
            type: "object",
            properties: {
              origin: { type: "string" },
              destination: { type: "string" }
            },
            required: ["origin", "destination"]
          },
          strict: true
        }
      },
      {
        type: "function",
        function: {
          name: "search_knowledge",
          description: "Search Kerala knowledge databases.",
          parameters: {
            type: "object",
            properties: {
              query: { type: "string" },
              category: { type: "string", enum: ["writer", "book", "history", "culture", "festival", "government_service", "food_knowledge", "education", "emergency", "general_kerala"] },
              limit: { type: "integer", minimum: 1, maximum: 8 }
            },
            required: ["query", "category"]
          },
          strict: true
        }
      }
    ];
  }

  async function executeTool(apiBase, name, args) {
    return jsonFetch(`${apiBase}/api/agent/tool`, {
      method: "POST",
      body: JSON.stringify({ name, arguments: args || {} })
    });
  }

  async function ask(options) {
    const cfg = { ...DEFAULTS, ...(options || {}) };
    const message = String(cfg.message || "").trim();
    if (!message) throw new Error("message is required");

    await ensurePuterAuth();

    // First pass: let the backend do cheap deterministic routing and provide
    // a small amount of likely context. The LLM can fetch more via tools.
    const ctx = await jsonFetch(`${cfg.apiBase}/api/agent/context`, {
      method: "POST",
      body: JSON.stringify({
        message,
        history: cfg.history || [],
        currentPlaceId: cfg.currentPlaceId || null,
        lastMatchedPlaceIds: cfg.lastMatchedPlaceIds || [],
        userLat: cfg.userLat || 0,
        userLng: cfg.userLng || 0,
        userLocationText: cfg.userLocationText || "",
      })
    });

    const messages = [
      { role: "system", content: ctx.systemPrompt },
      {
        role: "system",
        content: "Initial private app context (use as hints; call tools when more data is needed):\n" + JSON.stringify(compactInitialContext(ctx))
      },
      ...(ctx.history || []),
      { role: "user", content: message }
    ];

    let response = await global.puter.ai.chat(messages, {
      model: cfg.model,
      tools: makeTools(cfg.apiBase),
      max_tokens: cfg.maxTokens,
      temperature: cfg.temperature,
      reasoning_effort: cfg.reasoningEffort,
      normalize: true,
    });

    for (let round = 0; round < cfg.maxToolRounds; round++) {
      const assistantMessage = response?.message;
      const calls = assistantMessage?.tool_calls || [];
      if (!calls.length) break;

      // Preserve the exact assistant tool-call message so the next request has
      // the correct tool_call IDs.
      messages.push({
        role: "assistant",
        content: assistantMessage.content || "",
        tool_calls: calls,
      });

      for (const call of calls) {
        const name = call.function?.name;
        let args = {};
        try { args = JSON.parse(call.function?.arguments || "{}"); }
        catch (_) { args = {}; }

        const result = await executeTool(cfg.apiBase, name, args);
        messages.push({
          role: "tool",
          tool_call_id: call.id,
          content: JSON.stringify(result),
        });
      }

      response = await global.puter.ai.chat(messages, {
        model: cfg.model,
        tools: makeTools(cfg.apiBase),
        max_tokens: cfg.maxTokens,
        temperature: cfg.temperature,
        reasoning_effort: cfg.reasoningEffort,
        normalize: true,
      });
    }

    const answer = contentOf(response).trim();
    return {
      answer,
      response,
      context: ctx,
      puter: {
        model: cfg.model,
        signedIn: true,
        userPays: true,
      }
    };
  }

  async function signIn() {
    await ensurePuterAuth();
    return global.puter.auth?.getUser ? global.puter.auth.getUser() : null;
  }

  global.OctapusPuterV4 = { ask, signIn, makeTools };
})(window);
