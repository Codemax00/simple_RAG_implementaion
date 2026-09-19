/**
 * API MODULE: Handles all HTTP & Server-Sent Events (SSE) requests with the backend.
 */

export const Api = {
  async getModels() {
    const res = await fetch("/api/models");
    if (!res.ok) throw new Error("Failed to load models");
    return await res.json();
  },

  async selectModel(provider, modelName, temperature = 0.2) {
    const res = await fetch("/api/models/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider, model_name: modelName, temperature }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to switch model");
    }
    return await res.json();
  },

  async getDocuments() {
    const res = await fetch("/api/documents");
    if (!res.ok) throw new Error("Failed to load documents");
    return await res.json();
  },

  async uploadFileStreaming(file, onProgress) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/api/documents/upload", {
      method: "POST",
      body: formData,
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));
            if (onProgress) onProgress(data);
          } catch (e) {
            console.error("SSE JSON parse error:", e);
          }
        }
      }
    }
  },

  async streamChat(message, threadId, onChunk, signal = null) {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, thread_id: threadId }),
      signal,
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();

      for (const ev of events) {
        if (ev.startsWith("data: ")) {
          try {
            const payload = JSON.parse(ev.slice(6));
            if (onChunk) onChunk(payload);
          } catch (e) {
            console.error("Chat SSE parse error:", e);
          }
        }
      }
    }
  },

  async resetSession() {
    const res = await fetch("/api/chat/reset", { method: "POST" });
    return await res.json();
  },
};
