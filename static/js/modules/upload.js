/**
 * UPLOAD MODULE: Manages the document library drawer, drag & drop uploads,
 * and live visualization of the async chunking & embedding pipeline.
 */

import { Api } from "./api.js";

export class UploadManager {
  constructor() {
    this.btnOpen = document.getElementById("btnOpenDocs");
    this.drawer = document.getElementById("docDrawer");
    this.backdrop = document.getElementById("drawerBackdrop");
    this.btnClose = document.getElementById("btnCloseDrawer");
    this.badgeCount = document.getElementById("docBadgeCount");
    this.docList = document.getElementById("docList");
    this.dropzone = document.getElementById("dropzone");
    this.fileInput = document.getElementById("fileInput");

    // Live Pipeline Box
    this.pipelineBox = document.getElementById("pipelineBox");
    this.pipelineLog = document.getElementById("pipelineLog");
    this.pipelineBarFill = document.getElementById("pipelineBarFill");
    this.pipelineStatusText = document.getElementById("pipelineStatusText");

    this.bindEvents();
  }

  bindEvents() {
    this.btnOpen.addEventListener("click", () => this.open());
    this.btnClose.addEventListener("click", () => this.close());
    this.backdrop.addEventListener("click", () => this.close());

    this.dropzone.addEventListener("click", () => this.fileInput.click());
    this.fileInput.addEventListener("change", (e) => this.handleFiles(e.target.files));

    ["dragenter", "dragover"].forEach((ev) => {
      this.dropzone.addEventListener(ev, (e) => {
        e.preventDefault();
        this.dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((ev) => {
      this.dropzone.addEventListener(ev, (e) => {
        e.preventDefault();
        this.dropzone.classList.remove("dragover");
      });
    });

    this.dropzone.addEventListener("drop", (e) => {
      if (e.dataTransfer && e.dataTransfer.files.length > 0) {
        this.handleFiles(e.dataTransfer.files);
      }
    });
  }

  async load() {
    try {
      const data = await Api.getDocuments();
      this.badgeCount.textContent = `${data.total_chunks} chunks`;
      this.renderList(data.files);
    } catch (e) {
      console.error("Failed to load documents:", e);
    }
  }

  open() {
    this.drawer.classList.add("active");
    this.backdrop.classList.add("active");
    this.load();
  }

  close() {
    this.drawer.classList.remove("active");
    this.backdrop.classList.remove("active");
  }

  renderList(files) {
    this.docList.innerHTML = "";
    if (!files || files.length === 0) {
      this.docList.innerHTML = `<div style="color: var(--text-muted); font-size: 0.85rem; padding: 1rem 0;">No documents in library yet. Drop files above to index them.</div>`;
      return;
    }

    files.forEach((file) => {
      const item = document.createElement("div");
      item.className = "doc-item-card";

      item.innerHTML = `
        <div class="doc-item-info">
          <span class="doc-badge-ext">${file.extension || "TXT"}</span>
          <div class="doc-details">
            <div class="doc-name" title="${file.name}">${file.name}</div>
            <div class="doc-meta">${file.size_formatted}</div>
          </div>
        </div>
        <div class="doc-chunk-pill">
          ${file.is_indexed ? `${file.chunk_count} chunks` : "Pending"}
        </div>
      `;
      this.docList.appendChild(item);
    });
  }

  async handleFiles(fileList) {
    if (!fileList || fileList.length === 0) return;

    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      await this.uploadSingleStreaming(file);
    }

    await this.load();
  }

  async uploadSingleStreaming(file) {
    this.pipelineBox.classList.add("active");
    this.pipelineLog.innerHTML = "";
    this.pipelineBarFill.style.width = "15%";
    this.pipelineStatusText.textContent = `Uploading ${file.name}...`;

    const logEntry = (msg) => {
      const row = document.createElement("div");
      row.textContent = `• ${msg}`;
      this.pipelineLog.appendChild(row);
      this.pipelineLog.scrollTop = this.pipelineLog.scrollHeight;
    };

    try {
      await Api.uploadFileStreaming(file, (data) => {
        if (data.message) logEntry(data.message);

        if (data.stage === "start") {
          this.pipelineBarFill.style.width = "40%";
          this.pipelineStatusText.textContent = "Chunk 1 generated -> Streaming to Embedding...";
        } else if (data.stage === "done") {
          this.pipelineBarFill.style.width = "100%";
          this.pipelineStatusText.textContent = `Done! Indexed into ChromaDB.`;
        }
      });
    } catch (e) {
      logEntry(`Upload error: ${e.message}`);
    }
  }
}
