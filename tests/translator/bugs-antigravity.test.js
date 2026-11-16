// Real Antigravity-MITM requests (Gemini-internal: { request: { contents, ... } }) → OpenAI.
import { describe, it, expect } from "vitest";
import "./registerAll.js";
import { translateRequest, translateResponse, initState } from "../../open-sse/translator/index.js";
import { FORMATS } from "../../open-sse/translator/formats.js";
import { AntigravityExecutor } from "../../open-sse/executors/antigravity.js";
import { openaiToAntigravityRequest } from "../../open-sse/translator/request/openai-to-gemini.js";
import { ANTIGRAVITY_DEFAULT_SYSTEM } from "../../open-sse/config/appConstants.js";

const AG2O = (req) =>
  translateRequest(FORMATS.ANTIGRAVITY, FORMATS.OPENAI, "m", { request: req }, true, null, null);

describe("Antigravity → OpenAI", () => {
  // antigravity-to-openai.js — content with BOTH functionResponse and functionCall/text
  // previously returned toolResults early → dropped tool calls / text (fixed in #2225)
  it("functionResponse + functionCall in same content keeps both", () => {
    const out = AG2O({
      contents: [{
        role: "model",
        parts: [
          { functionResponse: { id: "c1", name: "prev", response: { result: "done" } } },
          { functionCall: { id: "c2", name: "next", args: {} } },
        ],
      }],
    });
    const json = JSON.stringify(out);
    expect(json, "functionCall lost when sharing content with functionResponse").toContain("\"next\"");
  });

  // antigravity-to-openai.js:167 — functionCall without id gets a random Date.now() id
  // KNOWN BUG: unstable id breaks matching with its functionResponse
  it("functionCall without id keeps a stable matchable id", () => {
    const out = AG2O({
      contents: [
        { role: "model", parts: [{ functionCall: { name: "search", args: { q: "x" } } }] },
        { role: "user", parts: [{ functionResponse: { name: "search", response: { result: "r" } } }] },
      ],
    });
    const asst = out.messages.find((m) => m.tool_calls);
    const tool = out.messages.find((m) => m.role === "tool");
    expect(tool?.tool_call_id, "id mismatch between call and response").toBe(asst?.tool_calls?.[0]?.id);
  });

  // antigravity-to-openai.js:144-147 — signature-only part handling (regression guard)
  it("signature-only part does not produce empty text", () => {
    const out = AG2O({
      contents: [{ role: "model", parts: [{ thoughtSignature: "sig", text: "" }] }],
    });
    const asst = out.messages.find((m) => m.role === "assistant");
    const content = asst?.content;
    const hasEmpty = Array.isArray(content)
      ? content.some((c) => c.type === "text" && c.text === "")
      : content === "";
    expect(hasEmpty, "empty text part emitted").toBe(false);
  });
});

describe("Antigravity → Claude", () => {
  it("tool call input_json_delta includes Anthropic index", () => {
    const state = initState(FORMATS.CLAUDE);
    const events = translateResponse(FORMATS.ANTIGRAVITY, FORMATS.CLAUDE, {
      response: {
        responseId: "resp-1",
        modelVersion: "gemini-pro-agent",
        candidates: [{
          content: {
            role: "model",
            parts: [{ functionCall: { name: "bash", args: { command: "git status" } } }],
          },
          finishReason: "STOP",
          index: 0,
        }],
      },
    }, state);

    const jsonDelta = events.find(
      (event) => event.type === "content_block_delta" && event.delta?.type === "input_json_delta"
    );
    expect(jsonDelta).toMatchObject({ index: expect.any(Number) });
    expect(JSON.parse(jsonDelta.delta.partial_json)).toEqual({ command: "git status" });
  });
});

describe("Antigravity executor", () => {
  it.each(["optional", "~optional"])("strips %s from nested tool schemas", (annotation) => {
    const out = new AntigravityExecutor().transformRequest("gemini-2.5-pro", {
      request: {
        contents: [{ role: "user", parts: [{ text: "hi" }] }],
        tools: [{
          functionDeclarations: [{
            name: "lookup",
            description: "Lookup a value",
            parameters: {
              type: "object",
              properties: {
                query: {
                  type: "string",
                  description: "Search query",
                  [annotation]: true,
                },
              },
            },
          }],
        }],
      },
    }, true, { projectId: "project-1", connectionId: "conn-1" });

    const query = out.request.tools[0].functionDeclarations[0].parameters.properties.query;
    expect(query).toEqual({ type: "string", description: "Search query" });
  });

  it("does not inject the legacy Antigravity default system prompt for Gemini-backed models", () => {
    const out = openaiToAntigravityRequest("gemini-3.5-flash-low", {
      messages: [
        { role: "system", content: "USER_SYSTEM_PROMPT" },
        { role: "user", content: "hello" },
      ],
    }, true, { projectId: "project-1", connectionId: "conn-1" });

    const system = JSON.stringify(out.request.systemInstruction);
    expect(system).toContain("USER_SYSTEM_PROMPT");
    expect(system).not.toContain(ANTIGRAVITY_DEFAULT_SYSTEM);
    expect(system).not.toContain("Please ignore the following [ignore]");
  });

  it("does not inject the legacy Antigravity default system prompt for Claude-backed models", () => {
    const out = openaiToAntigravityRequest("claude-opus-4-6-thinking", {
      messages: [
        { role: "system", content: "USER_SYSTEM_PROMPT" },
        { role: "user", content: "hello" },
      ],
    }, true, { projectId: "project-1", connectionId: "conn-1" });

    const system = JSON.stringify(out.request.systemInstruction);
    expect(system).toContain("USER_SYSTEM_PROMPT");
    expect(system).not.toContain(ANTIGRAVITY_DEFAULT_SYSTEM);
    expect(system).not.toContain("Please ignore the following [ignore]");
  });

  it("strips Hermes Agent identity in systemInstruction to bypass Antigravity 429 filter (#3358)", () => {
    const executor = new AntigravityExecutor();
    const out = executor.transformRequest("gemini-2.5-pro", {
      request: {
        systemInstruction: {
          parts: [
            { text: "You are Hermes Agent, an intelligent AI assistant created by Nous Research. Be helpful." }
          ]
        },
        contents: [{ role: "user", parts: [{ text: "hi" }] }]
      }
    }, true, { projectId: "project-1", connectionId: "conn-1" });

    expect(out.request.systemInstruction.parts[0].text).toBe(" Be helpful.");
  });

  it("clears Hermes Agent identity through OpenAI -> Antigravity request translation (#3358)", () => {
    const out = openaiToAntigravityRequest("gemini-2.5-pro", {
      messages: [
        { role: "system", content: "You are Hermes Agent, an intelligent AI assistant created by Nous Research." },
        { role: "user", content: "hello" }
      ]
    }, true, { projectId: "project-1", connectionId: "conn-1" });

    const executor = new AntigravityExecutor();
    const transformed = executor.transformRequest("gemini-2.5-pro", out, true, { projectId: "project-1", connectionId: "conn-1" });
    expect(transformed.request.systemInstruction.parts[0].text).toBe("");
  });

  it("rewrites competing-client branding (Zed Claude SDK and OpenCode) in systemInstruction", () => {
    const executor = new AntigravityExecutor();
    const out = executor.transformRequest("gemini-2.5-pro", {
      request: {
        systemInstruction: {
          parts: [
            { text: "You are a Claude agent, built on Anthropic's Claude Agent SDK. OpenCode is great." }
          ]
        },
        contents: [{ role: "user", parts: [{ text: "hi" }] }]
      }
    }, true, { projectId: "project-1", connectionId: "conn-1" });

    expect(out.request.systemInstruction.parts[0].text).toBe(" Antigravity is great.");
  });
  it.each(["gemini-3.8-flash-high", "claude-sonnet-5-5-high"])("rewrites the OpenClaw identity sentence in systemInstruction for %s", (model) => {
    const out = new AntigravityExecutor().transformRequest(model, {
      request: {
        systemInstruction: {
          parts: [
            { text: "Preamble.\nYOU ARE A PERSONAL ASSISTANT RUNNING INSIDE OPENCLAW." },
            { text: "You are a personal assistant running inside OpenClaw" },
            { text: "Postamble." },
          ],
        },
        contents: [{ role: "user", parts: [{ text: "hi" }] }],
      },
    }, true, {});

    expect(out.request.systemInstruction.parts.map((part) => part.text)).toEqual([
      "Preamble.\nYou are a personal assistant.",
      "You are a personal assistant running inside OpenClaw",
      "Postamble.",
    ]);
  });

  it("does not rewrite the no-period variant or user messages", () => {
    const userText = "You are a personal assistant running inside OpenClaw.";
    const body = openaiToAntigravityRequest("gemini-3.8-flash-high", {
      messages: [
        { role: "system", content: "Keep this.\nYou are a personal assistant running inside OpenClaw" },
        { role: "user", content: userText },
      ],
    }, true);
    const out = new AntigravityExecutor().transformRequest("gemini-3.8-flash-high", body, true, {});
    const systemText = out.request.systemInstruction.parts.map((part) => part.text).join("\n");
    const contentsText = out.request.contents.flatMap((content) => content.parts || []).map((part) => part.text || "").join("\n");

    expect(systemText).toContain("You are a personal assistant running inside OpenClaw");
    expect(systemText).not.toContain("You are a personal assistant.");
    expect(contentsText).toContain(userText);
  });
  it("rewrites the coding-task false-429 phrase across system-prompt whitespace", () => {
    const userText = "Please solve coding tasks. Be concise, practical, and tool-aware.";
    const body = openaiToAntigravityRequest("gemini-3.8-flash-high", {
      messages: [
        {
          role: "system",
          content: "You are pair programming with a user to solve coding tasks.\nBe concise, practical, and tool-aware.\nKeep this instruction.",
        },
        { role: "user", content: userText },
      ],
    }, true);
    const out = new AntigravityExecutor().transformRequest("gemini-3.8-flash-high", body, true, {});
    const systemText = out.request.systemInstruction.parts.map((part) => part.text).join("\n");
    const contentsText = out.request.contents.flatMap((content) => content.parts || []).map((part) => part.text || "").join("\n");

    expect(systemText).toContain("work on their coding tasks. Keep responses concise and practical, and use the available tools when they help.");
    expect(systemText).not.toContain("solve coding tasks.");
    expect(contentsText).toContain(userText);
  });
});

