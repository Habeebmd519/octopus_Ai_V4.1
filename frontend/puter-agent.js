/*
 * Octapus AI V5 - Puter.js User-Pays Agent
 *
 * Required before this file:
 *   <script src="https://js.puter.com/v2/"></script>
 *
 * Architecture:
 *
 *   Octapus frontend
 *          ↓
 *      Puter AI
 *          ↓
 *   Octapus V4 backend tools
 *          ↓
 *   Firebase / Places / Tavily / OSM / Travel
 *
 * Puter owns the LLM request.
 * The Flask backend owns private application data and tools.
 */

(function (global) {
  "use strict";

  // ------------------------------------------------------------
  // DEFAULT CONFIG
  // ------------------------------------------------------------

  const DEFAULTS = {
    apiBase: "",
    model: "gpt-5.6-luna",

    // Maximum number of tool → AI → tool rounds.
    maxToolRounds: 4,

    // Maximum output tokens.
    maxTokens: 1400,

    // IMPORTANT:
    // Do NOT send temperature.
    // GPT-5.6 Luna currently rejects it in this integration.

    reasoningEffort: "medium",
  };


  // ------------------------------------------------------------
  // RESPONSE HELPERS
  // ------------------------------------------------------------

  function contentOf(response) {
    if (!response) {
      return "";
    }

    if (typeof response === "string") {
      return response;
    }

    // Normalized Puter response.
    if (response.message?.content) {
      if (typeof response.message.content === "string") {
        return response.message.content;
      }

      if (Array.isArray(response.message.content)) {
        return response.message.content
          .map(item => item?.text || "")
          .join("");
      }
    }

    // Fallback response formats.
    if (typeof response.text === "string") {
      return response.text;
    }

    if (typeof response.content === "string") {
      return response.content;
    }

    return "";
  }


  // ------------------------------------------------------------
  // BACKEND FETCH
  // ------------------------------------------------------------

  async function jsonFetch(url, options = {}) {
    const controller = new AbortController();

    const timer = setTimeout(() => {
      controller.abort();
    }, 30000);

    try {
      const response = await fetch(url, {
        method: "GET",

        headers: {
          "Content-Type": "application/json",
          ...(options.headers || {})
        },

        ...options,

        signal: controller.signal
      });

      const text = await response.text();

      let data;

      try {
        data = JSON.parse(text);
      } catch (_) {
        data = {
          raw: text
        };
      }

      if (!response.ok) {
        throw new Error(
          data?.message ||
          data?.error ||
          data?.raw ||
          `HTTP ${response.status}`
        );
      }

      return data;

    } catch (error) {

      if (error?.name === "AbortError") {
        throw new Error("Backend request timed out.");
      }

      throw error;

    } finally {
      clearTimeout(timer);
    }
  }


  // ------------------------------------------------------------
  // PUTER AUTHENTICATION
  // ------------------------------------------------------------

  async function ensurePuterAuth() {

    if (!global.puter) {
      throw new Error(
        "Puter.js is not loaded. Check the Puter script tag."
      );
    }

    if (!global.puter.auth) {
      throw new Error(
        "Puter authentication API is unavailable."
      );
    }

    // Already signed in.
    if (global.puter.auth.isSignedIn()) {
      return true;
    }

    /*
     * IMPORTANT:
     *
     * This opens the normal Puter login popup.
     *
     * The frontend should call this from a user action:
     * - Connect Puter button
     * - Send button
     *
     * This restores the old V4 behaviour.
     */
    await global.puter.auth.signIn();

    // Verify authentication actually completed.
    if (!global.puter.auth.isSignedIn()) {
      throw new Error(
        "Puter sign-in did not complete."
      );
    }

    return true;
  }


  // ------------------------------------------------------------
  // INITIAL BACKEND CONTEXT
  // ------------------------------------------------------------

  function compactInitialContext(ctx) {

    return {
      masterIntent: ctx?.masterIntent,
      intent: ctx?.intent,

      userLocation: ctx?.userLocation,

      currentPlace: ctx?.currentPlace,

      initialResults: ctx?.initialResults,

      live: ctx?.live,

      placeCount: ctx?.placeCount,

      imageCount: ctx?.imageCount
    };
  }


  // ------------------------------------------------------------
  // PUTER TOOLS
  // ------------------------------------------------------------

  function makeTools(apiBase) {

    return [

      // --------------------------------------------------------
      // SEARCH PLACES
      // --------------------------------------------------------

      {
        type: "function",

        function: {
          name: "search_places",

          description:
            "Search the private KeralaTour database of 18,000+ Kerala places. Use this for destinations, attractions, restaurants, hotels, viewpoints, waterfalls, temples, beaches and other places.",

          parameters: {
            type: "object",

            properties: {

              query: {
                type: "string"
              },

              limit: {
                type: "integer",
                minimum: 1,
                maximum: 12
              }

            },

            required: [
              "query"
            ]
          },

          strict: false
        }
      },


      // --------------------------------------------------------
      // GET PLACE
      // --------------------------------------------------------

      {
        type: "function",

        function: {
          name: "get_place",

          description:
            "Get detailed information about one specific KeralaTour place using its place ID or name.",

          parameters: {
            type: "object",

            properties: {

              place_id: {
                type: "string"
              },

              name: {
                type: "string"
              }

            }
          },

          strict: false
        }
      },


      // --------------------------------------------------------
      // SEARCH SERVICES
      // --------------------------------------------------------

      {
        type: "function",

        function: {
          name: "search_services",

          description:
            "Find practical nearby services using OpenStreetMap, including food, accommodation, healthcare, transport, emergency services, money and fuel.",

          parameters: {
            type: "object",

            properties: {

              query: {
                type: "string"
              },

              category: {
                type: "string",

                enum: [
                  "food",
                  "osm_stay",
                  "osm_health",
                  "osm_transport",
                  "osm_emergency",
                  "osm_money",
                  "osm_fuel"
                ]
              },

              limit: {
                type: "integer",
                minimum: 1,
                maximum: 12
              },

              lat: {
                type: "number"
              },

              lng: {
                type: "number"
              }

            },

            required: [
              "query"
            ]
          },

          strict: false
        }
      },


      // --------------------------------------------------------
      // LIVE WEB SEARCH
      // --------------------------------------------------------

      {
        type: "function",

        function: {
          name: "live_search",

          description:
            "Search current web information using Tavily. Use this for current news, current events, latest information, changing prices, current schedules and other information that may have changed.",

          parameters: {
            type: "object",

            properties: {

              query: {
                type: "string"
              },

              limit: {
                type: "integer",
                minimum: 1,
                maximum: 5
              }

            },

            required: [
              "query"
            ]
          },

          strict: false
        }
      },


      // --------------------------------------------------------
      // TRAVEL
      // --------------------------------------------------------

      {
        type: "function",

        function: {
          name: "travel_info",

          description:
            "Get route distance and estimated travel duration between an origin and destination.",

          parameters: {
            type: "object",

            properties: {

              origin: {
                type: "string"
              },

              destination: {
                type: "string"
              }

            },

            required: [
              "origin",
              "destination"
            ]
          },

          strict: false
        }
      },


      // --------------------------------------------------------
      // KNOWLEDGE
      // --------------------------------------------------------

      {
        type: "function",

        function: {
          name: "search_knowledge",

          description:
            "Search Octapus Kerala knowledge databases for writers, books, history, culture, festivals, government services, food knowledge, education, emergencies and general Kerala knowledge.",

          parameters: {
            type: "object",

            properties: {

              query: {
                type: "string"
              },

              category: {
                type: "string",

                enum: [
                  "writer",
                  "book",
                  "history",
                  "culture",
                  "festival",
                  "government_service",
                  "food_knowledge",
                  "education",
                  "emergency",
                  "general_kerala"
                ]
              },

              limit: {
                type: "integer",
                minimum: 1,
                maximum: 8
              }

            },

            required: [
              "query",
              "category"
            ]
          },

          strict: false
        }
      }

    ];
  }


  // ------------------------------------------------------------
  // EXECUTE BACKEND TOOL
  // ------------------------------------------------------------

  async function executeTool(apiBase, name, args) {

    if (!apiBase) {
      throw new Error(
        "Octapus backend URL is missing."
      );
    }

    return jsonFetch(
      `${apiBase}/api/agent/tool`,
      {
        method: "POST",

        body: JSON.stringify({
          name: name,
          arguments: args || {}
        })
      }
    );
  }


  // ------------------------------------------------------------
  // ASK PUTER
  // ------------------------------------------------------------

  async function ask(options = {}) {

    const cfg = {
      ...DEFAULTS,
      ...options
    };

    const message = String(
      cfg.message || ""
    ).trim();

    if (!message) {
      throw new Error(
        "message is required"
      );
    }


    // ----------------------------------------------------------
    // AUTH
    // ----------------------------------------------------------

    await ensurePuterAuth();


    // ----------------------------------------------------------
    // BACKEND CONTEXT
    // ----------------------------------------------------------

    if (!cfg.apiBase) {
      throw new Error(
        "Octapus API base URL is missing."
      );
    }

    const ctx = await jsonFetch(
      `${cfg.apiBase}/api/agent/context`,
      {
        method: "POST",

        body: JSON.stringify({

          message,

          history:
            cfg.history || [],

          currentPlaceId:
            cfg.currentPlaceId || null,

          lastMatchedPlaceIds:
            cfg.lastMatchedPlaceIds || [],

          userLat:
            cfg.userLat || 0,

          userLng:
            cfg.userLng || 0,

          userLocationText:
            cfg.userLocationText || ""
        })
      }
    );


    // ----------------------------------------------------------
    // BUILD MESSAGES
    // ----------------------------------------------------------

    const messages = [

      {
        role: "system",

        content:
          (ctx.systemPrompt ||
            "You are Octapus AI, a Kerala-first AI assistant.") +
          "\n\nIMPORTANT PRODUCT INFORMATION:\n" +
          "You are the AI reasoning layer of Octapus AI.\n" +
          "The configured model for this conversation is GPT-5.6 Luna through Puter AI.\n" +
          "If the user asks which model you use, answer clearly: " +
          "I use GPT-5.6 Luna through Puter AI."
      },

      {
        role: "system",

        content:
          "Initial private app context. Treat this as application data and hints. Use the available tools whenever additional grounded information is needed.\n\n" +
          JSON.stringify(
            compactInitialContext(ctx)
          )
      },

      ...(ctx.history || []),

      {
        role: "user",

        content: message
      }

    ];


    // ----------------------------------------------------------
    // FIRST PUTER AI CALL
    // ----------------------------------------------------------

    let response =
      await global.puter.ai.chat(
        messages,
        {

          model:
            cfg.model,

          tools:
            makeTools(
              cfg.apiBase
            ),

          max_tokens:
            cfg.maxTokens,

          reasoning_effort:
            cfg.reasoningEffort,

          normalize:
            true
        }
      );


    // ----------------------------------------------------------
    // TOOL LOOP
    // ----------------------------------------------------------

    for (
      let round = 0;
      round < cfg.maxToolRounds;
      round++
    ) {

      const assistantMessage =
        response?.message;

      const calls =
        assistantMessage?.tool_calls || [];


      // No tool call = final answer.
      if (!calls.length) {
        break;
      }


      // --------------------------------------------------------
      // PRESERVE ASSISTANT TOOL CALL
      // --------------------------------------------------------

      messages.push(assistantMessage);


      // --------------------------------------------------------
      // EXECUTE EACH TOOL
      // --------------------------------------------------------

      for (const call of calls) {

        const name =
          call?.function?.name;

        if (!name) {
          continue;
        }


        let args = {};

        try {

          const rawArguments =
            call?.function?.arguments;

          if (
            typeof rawArguments === "string"
          ) {

            args =
              JSON.parse(
                rawArguments || "{}"
              );

          } else if (
            rawArguments &&
            typeof rawArguments === "object"
          ) {

            args =
              rawArguments;
          }

        } catch (error) {

          console.warn(
            "Could not parse Puter tool arguments:",
            error
          );

          args = {};
        }


        let result;

        try {

          result =
            await executeTool(
              cfg.apiBase,
              name,
              args
            );

        } catch (toolError) {

          /*
           * Give the model a tool error instead of killing
           * the entire conversation.
           */

          result = {
            ok: false,

            error:
              toolError?.message ||
              String(toolError)
          };
        }


        messages.push({

          role: "tool",

          tool_call_id:
            call.id,

          content:
            JSON.stringify(
              result
            )

        });
      }


      // --------------------------------------------------------
      // SECOND / NEXT PUTER AI CALL
      // --------------------------------------------------------

      response =
        await global.puter.ai.chat(
          messages,
          {

            model:
              cfg.model,

            tools:
              makeTools(
                cfg.apiBase
              ),

            max_tokens:
              cfg.maxTokens,

            reasoning_effort:
              cfg.reasoningEffort,

            normalize:
              true
          }
        );
    }


    // ----------------------------------------------------------
    // FINAL ANSWER
    // ----------------------------------------------------------

    const answer =
      contentOf(response)
        .trim();


    if (!answer) {

      throw new Error(
        "Puter returned an empty answer."
      );
    }


    // ----------------------------------------------------------
    // RESULT
    // ----------------------------------------------------------

    return {

      answer,

      response,

      context:
        ctx,

      puter: {

        model:
          cfg.model,

        signedIn:
          true,

        userPays:
          true
      }

    };
  }


  // ------------------------------------------------------------
  // MANUAL SIGN IN
  // ------------------------------------------------------------

  async function signIn() {

    await ensurePuterAuth();

    if (
      global.puter.auth?.getUser
    ) {

      return (
        await global.puter.auth.getUser()
      );
    }

    return null;
  }


  // ------------------------------------------------------------
  // SIGN-IN STATUS
  // ------------------------------------------------------------

  function isSignedIn() {

    try {

      return Boolean(
        global.puter &&
        global.puter.auth &&
        global.puter.auth.isSignedIn()
      );

    } catch (_) {

      return false;
    }
  }


  // ------------------------------------------------------------
  // PUBLIC API
  // ------------------------------------------------------------

  global.OctapusPuterV4 = {

    ask,

    signIn,

    isSignedIn,

    makeTools
  };

})(window);