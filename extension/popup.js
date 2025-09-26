const BACKEND = "https://mykereta-api-487fe4f80f73.herokuapp.com";

const $ = (s)=>document.querySelector(s);
const $status = $("#status"), $login = $("#login"), $fill = $("#fill");

// status
chrome.runtime.sendMessage({ type: "GET_STATUS" }, (res) => {
  if (res?.ok) {
    $status.textContent = "Signed in ✓";
    $login.style.display = "none";
  } else {
    $status.textContent = "Not signed in";
  }
});

// login starts normal web flow (cookie set on backend domain)
$login.onclick = ()=> chrome.tabs.create({ url: `${BACKEND}/api/auth/google/start` });

// Fill current tab
$fill.onclick = async () => {
  // 1) get data in background (includes credentials)
  chrome.runtime.sendMessage({ type: "GET_AUTOFILL" }, async (res) => {
    if (!res?.ok) { $status.textContent = "Auth required"; return; }

    // 2) inject content script (only when needed)
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });

    // 3) send data to the page
    chrome.tabs.sendMessage(tab.id, { type: "FILL_NOW", payload: res.data }, (r) => {
      $status.textContent = r?.count ? `Filled ${r.count} fields` : "No matching fields";
    });
  });
};