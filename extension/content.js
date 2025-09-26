// map normalized keys -> common selectors/keywords on real forms
const KEYWORDS = {
    fullName: ["name", "full name", "nama", "applicant name"],
    email: ["email", "e-mail"],
    icNo: ["ic", "nric", "identity", "id number", "no. kp", "kad pengenalan"],
    phone: ["phone", "mobile", "telefon"],
    address1: ["address 1", "alamat 1", "address line 1", "street"],
    address2: ["address 2", "alamat 2", "address line 2"],
    city: ["city", "bandar"],
    state: ["state", "negeri"],
    postcode: ["postcode", "zip", "poskod"],
    licenseNo: ["license", "lesen", "driving license"],
    vehiclePlate: ["plate", "registration", "no plate", "no. pendaftaran"],
    vehicleChassis: ["chassis", "chasis", "vin"],
    vehicleBrand: ["brand", "make"],
    vehicleModel: ["model"],
    vehicleYear: ["year"],
    insurancePolicyNo: ["policy", "insurance policy"],
    roadTaxExpiry: ["road tax", "expiry", "tamat tempoh"]
  };
  
  // try fill by label[for], name, id, placeholder, aria-label, autocomplete
  function score(el, words) {
    const hay = [
      el.name, el.id, el.placeholder,
      el.getAttribute("aria-label"),
      el.getAttribute("autocomplete"),
      el.closest("label")?.textContent,
      document.querySelector(`label[for="${el.id}"]`)?.textContent
    ].join(" ").toLowerCase();
    return words.some(w => hay.includes(w)) ? 1 : 0;
  }
  
  function fill(fields) {
    const inputs = Array.from(document.querySelectorAll("input, textarea, select"));
    let filled = 0;
  
    for (const [key, value] of Object.entries(fields)) {
      if (!value) continue;
      const words = KEYWORDS[key] || [];
      // pick the first best match
      const candidate = inputs.find(el => score(el, words));
      if (candidate) {
        candidate.focus();
        candidate.value = value;
        candidate.dispatchEvent(new Event("input", { bubbles: true }));
        candidate.dispatchEvent(new Event("change", { bubbles: true }));
        filled++;
      }
    }
    return filled;
  }
  
  // listen for background sending data to fill
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "FILL_NOW" && msg.payload?.fields) {
      const n = fill(msg.payload.fields);
      chrome.runtime.sendMessage({ type: "FILL_RESULT", count: n });
    }
  });