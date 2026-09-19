/**
 * MAIN ENTRYPOINT: Bootstraps and coordinates all frontend modules.
 */

import { ModelManager } from "./modules/models.js";
import { UploadManager } from "./modules/upload.js";
import { ChatManager } from "./modules/chat.js";

window.addEventListener("DOMContentLoaded", async () => {
  const chat = new ChatManager();
  const upload = new UploadManager();
  const models = new ModelManager((provider, modelName) => {
    chat.appendNotice(`Active model switched to ${modelName} (${provider.toUpperCase()})`);
  });

  // Initialize initial data
  await Promise.all([models.load(), upload.load()]);
});
