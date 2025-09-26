const BACKEND = "https://mykereta-api-487fe4f80f73.herokuapp.com";

async function fetchAutofill() {
  const res = await fetch(`${BACKEND}/api/extension/autofill`, {
    credentials: "include"
  });
  if (!res.ok) throw new Error("Not authenticated");
  return res.json();
}

async function sendVehicle(payload) {
  await fetch("https://mykereta-api-487fe4f80f73.herokuapp.com/api/extension/vehicle", {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-extension-key": "demo" },
    body: JSON.stringify(payload),
  });
}

// Receive requests from popup or content script
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    try {
      if (msg.type === "GET_STATUS") {
        const me = await fetch(`${BACKEND}/api/auth/me`, { credentials: "include" });
        sendResponse({ ok: me.ok });
      } else if (msg.type === "GET_AUTOFILL") {
        const data = await fetchAutofill();
        sendResponse({ ok: true, data });
      }
    } catch (e) {
      sendResponse({ ok: false, error: e.message });
    }
  })();
  
  return true; // keep the channel open for async response
});