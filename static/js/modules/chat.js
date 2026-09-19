/**
 * CHAT MODULE: Handles real-time token streaming, user messages, tool badges, and markdown rendering.
 */

import { Api } from "./api.js";

export class ChatManager {
  constructor() {
    this.threadId = null;
    this.isGenerating = false;
    this.abortController = null;

    // DOM Elements
    this.viewport = document.getElementById("chatViewport");
    this.container = document.getElementById("messagesContainer");
    this.welcomeHero = document.getElementById("welcomeHero");
    this.input = document.getElementById("chatInput");
    this.btnSend = document.getElementById("btnSend");
    this.btnStop = document.getElementById("btnStop");
    this.btnReset = document.getElementById("btnReset");

    this.bindEvents();
  }

  bindEvents() {
    this.input.addEventListener("input", () => {
      this.input.style.height = "auto";
      this.input.style.height = Math.min(this.input.scrollHeight, 180) + "px";
    });

    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.send();
      }
    });

    // Stop shortcut (Esc)
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.isGenerating) {
        e.preventDefault();
        this.stop();
      }
    });

    this.btnSend.addEventListener("click", () => this.send());
    if (this.btnStop) {
      this.btnStop.addEventListener("click", () => this.stop());
    }
    this.btnReset.addEventListener("click", () => this.reset());

    // Prompt Cards Click
    document.querySelectorAll(".hero-card").forEach((card) => {
      card.addEventListener("click", () => {
        const query = card.getAttribute("data-query");
        if (query) {
          this.input.value = query;
          this.send();
        }
      });
    });
  }

  stop() {
    if (this.isGenerating && this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }

  async send() {
    const text = this.input.value.trim();
    if (!text || this.isGenerating) return;

    if (this.welcomeHero) {
      this.welcomeHero.style.display = "none";
    }

    this.input.value = "";
    this.input.style.height = "auto";
    this.abortController = new AbortController();
    this.setGenerating(true);

    // Render User Message
    this.appendUserMessage(text);

    // Prepare Assistant Message Shell
    const { row, bubble, toolsContainer } = this.createAssistantShell();
    this.container.appendChild(row);
    this.scrollToBottom();

    const cursor = document.createElement("span");
    cursor.className = "streaming-cursor";
    bubble.appendChild(cursor);

    let rawMarkdown = "";

    try {
      await Api.streamChat(
        text,
        this.threadId,
        (payload) => {
          if (payload.type === "tool") {
            this.appendToolBadge(toolsContainer, payload.name, payload.args);
          } else if (payload.type === "token") {
            rawMarkdown += payload.content;
            cursor.remove();
            bubble.innerHTML = this.renderMarkdown(rawMarkdown);
            bubble.appendChild(cursor);
            this.scrollToBottom();
          } else if (payload.type === "done") {
            cursor.remove();
          } else if (payload.type === "error") {
            cursor.remove();
            bubble.innerHTML += `<div style="color: var(--accent-rose); margin-top: 0.5rem;">[Error: ${payload.error}]</div>`;
          }
        },
        this.abortController.signal
      );
    } catch (e) {
      cursor.remove();
      if (e.name === "AbortError") {
        bubble.innerHTML += `<div style="color: var(--text-muted); font-size: 0.8rem; margin-top: 0.5rem; font-style: italic;">[Generation stopped by user]</div>`;
      } else {
        bubble.innerHTML += `<div style="color: var(--accent-rose); margin-top: 0.5rem;">[Network error: ${e.message}]</div>`;
      }
    } finally {
      cursor.remove();
      this.abortController = null;
      this.setGenerating(false);
      this.input.focus();
    }
  }

  appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user-row";

    const avatar = document.createElement("div");
    avatar.className = "avatar-icon";
    avatar.textContent = "U";

    const wrapper = document.createElement("div");
    wrapper.className = "message-content-wrapper";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = text;

    wrapper.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(wrapper);
    this.container.appendChild(row);
    this.scrollToBottom();
  }

  createAssistantShell() {
    const row = document.createElement("div");
    row.className = "message-row assistant-row";

    const avatar = document.createElement("div");
    avatar.className = "avatar-icon";
    avatar.textContent = "AI";

    const wrapper = document.createElement("div");
    wrapper.className = "message-content-wrapper";

    const toolsContainer = document.createElement("div");
    toolsContainer.className = "tool-calls-container";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    wrapper.appendChild(toolsContainer);
    wrapper.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(wrapper);

    return { row, bubble, toolsContainer };
  }

  appendToolBadge(container, toolName, args) {
    const badge = document.createElement("div");
    badge.className = "tool-call-banner";

    let queryStr = "";
    if (typeof args === "object" && args !== null) {
      queryStr = args.query || args.expression || JSON.stringify(args);
    } else {
      queryStr = String(args);
    }

    badge.innerHTML = `
      <span class="tool-spinner"></span>
      <span>${toolName}(${queryStr})</span>
    `;
    container.appendChild(badge);
    this.scrollToBottom();
  }

  appendNotice(msg) {
    const notice = document.createElement("div");
    notice.style.textAlign = "center";
    notice.style.fontSize = "0.78rem";
    notice.style.color = "var(--text-muted)";
    notice.style.padding = "0.5rem 0";
    notice.textContent = `[System] ${msg}`;
    this.container.appendChild(notice);
    this.scrollToBottom();
  }

  async reset() {
    try {
      const data = await Api.resetSession();
      this.threadId = data.thread_id;
      this.container.innerHTML = "";
      if (this.welcomeHero) {
        this.welcomeHero.style.display = "block";
      }
      this.appendNotice("Conversation history cleared. Fresh session started.");
    } catch (e) {
      console.error("Failed to reset session:", e);
    }
  }

  setGenerating(isGenerating) {
    this.isGenerating = isGenerating;
    if (isGenerating) {
      if (this.btnSend) this.btnSend.style.display = "none";
      if (this.btnStop) this.btnStop.style.display = "flex";
    } else {
      if (this.btnStop) this.btnStop.style.display = "none";
      if (this.btnSend) {
        this.btnSend.style.display = "flex";
        this.btnSend.disabled = false;
      }
    }
  }

  scrollToBottom() {
    this.viewport.scrollTop = this.viewport.scrollHeight;
  }

  renderMarkdown(text) {
    if (!text) return "";
    let html = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Code blocks ```python ... ```
    html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Inline code `...`
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Bold **...**
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // Italic *...*
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");

    // Paragraphs & Lists
    const paragraphs = html.split("\n\n");
    return paragraphs
      .map((p) => {
        p = p.trim();
        if (p.startsWith("<pre>") && p.endsWith("</pre>")) return p;
        if (p.startsWith("&gt;")) return `<blockquote>${p.slice(4).trim()}</blockquote>`;
        if (p.startsWith("- ") || p.startsWith("* ") || p.startsWith("• ")) {
          const items = p.split("\n").map((li) => `<li>${li.replace(/^[-*•]\s*/, "")}</li>`).join("");
          return `<ul>${items}</ul>`;
        }
        return `<p>${p.replace(/\n/g, "<br>")}</p>`;
      })
      .join("");
  }
}
