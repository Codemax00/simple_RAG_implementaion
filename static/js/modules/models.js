/**
 * MODELS MODULE: Manages active model display and the interactive model selector modal.
 */

import { Api } from "./api.js";

export class ModelManager {
  constructor(onModelChanged) {
    this.onModelChanged = onModelChanged;
    this.activeModel = null;
    this.providers = [];
    this.selectedProvider = null;

    // DOM Elements
    this.btnOpen = document.getElementById("btnOpenModel");
    this.statusDot = document.getElementById("modelStatusDot");
    this.label = document.getElementById("modelLabel");
    this.modal = document.getElementById("modelModal");
    this.btnClose = document.getElementById("btnCloseModal");
    this.btnCancel = document.getElementById("btnCancelModal");
    this.btnConfirm = document.getElementById("btnConfirmModel");
    this.cardsContainer = document.getElementById("providerCardsContainer");

    this.bindEvents();
  }

  bindEvents() {
    this.btnOpen.addEventListener("click", () => this.open());
    this.btnClose.addEventListener("click", () => this.close());
    this.btnCancel.addEventListener("click", () => this.close());
    this.btnConfirm.addEventListener("click", () => this.apply());
  }

  async load() {
    try {
      const data = await Api.getModels();
      this.activeModel = data.active;
      this.providers = data.providers;
      this.updateHeader();
    } catch (e) {
      console.error("Failed to load models:", e);
    }
  }

  updateHeader() {
    if (!this.activeModel) return;
    const prov = this.activeModel.provider.toUpperCase();
    const name = this.activeModel.model_name;
    this.label.textContent = `${name} (${prov})`;

    const provObj = this.providers.find((p) => p.id === this.activeModel.provider);
    this.statusDot.className = "status-dot";
    if (provObj) {
      if (provObj.status === "ready") this.statusDot.classList.add("green");
      else if (provObj.status === "needs_key") this.statusDot.classList.add("amber");
      else this.statusDot.classList.add("red");
    }
  }

  open() {
    this.renderCards();
    this.modal.classList.add("active");
  }

  close() {
    this.modal.classList.remove("active");
  }

  renderCards() {
    this.cardsContainer.innerHTML = "";
    this.selectedProvider = this.activeModel.provider;

    this.providers.forEach((prov) => {
      const card = document.createElement("div");
      card.className = `provider-card ${prov.id === this.selectedProvider ? "selected" : ""}`;

      let dotClass = prov.status === "ready" ? "green" : prov.status === "needs_key" ? "amber" : "red";

      card.innerHTML = `
        <div class="provider-card-header">
          <div class="provider-title-group">
            <span class="status-dot ${dotClass}"></span>
            <span>${prov.name}</span>
          </div>
          <span style="font-size: 0.75rem; color: var(--text-muted);">${prov.status_message}</span>
        </div>
        <div class="provider-desc">${prov.description}</div>
        <select class="model-select-dropdown" id="sel-${prov.id}">
          ${prov.models.map((m) => `<option value="${m}" ${m === this.activeModel.model_name ? "selected" : ""}>${m}</option>`).join("")}
        </select>
      `;

      card.addEventListener("click", (e) => {
        if (e.target.tagName !== "SELECT") {
          document.querySelectorAll(".provider-card").forEach((c) => c.classList.remove("selected"));
          card.classList.add("selected");
          this.selectedProvider = prov.id;
        }
      });

      this.cardsContainer.appendChild(card);
    });
  }

  async apply() {
    if (!this.selectedProvider) return;
    const dropdown = document.getElementById(`sel-${this.selectedProvider}`);
    const chosenModel = dropdown ? dropdown.value : "";

    this.btnConfirm.textContent = "Switching...";
    this.btnConfirm.disabled = true;

    try {
      await Api.selectModel(this.selectedProvider, chosenModel);
      await this.load();
      this.close();
      if (this.onModelChanged) {
        this.onModelChanged(this.selectedProvider, chosenModel);
      }
    } catch (e) {
      alert("Error switching model: " + e.message);
    } finally {
      this.btnConfirm.textContent = "Save & Apply";
      this.btnConfirm.disabled = false;
    }
  }
}
