/* Clerk session helper. Only the publishable key is loaded from the server. */
(function () {
  const originalFetch = window.fetch.bind(window);
  // True only when the server has login switched on (it sends a Clerk key).
  let signInRequired = false;

  async function loadConfig() {
    const response = await originalFetch("/public-config");
    if (!response.ok) return {};
    return response.json();
  }

  function sameOrigin(input) {
    if (typeof input !== "string") return false;
    if (input.startsWith("/")) return true;
    try {
      return new URL(input, window.location.origin).origin === window.location.origin;
    } catch (_err) {
      return false;
    }
  }

  window.fetch = async function (input, init) {
    const next = init ? { ...init } : {};
    if (sameOrigin(input) && window.Clerk && window.Clerk.session) {
      const token = await window.Clerk.session.getToken();
      if (token) {
        const headers = new Headers(next.headers || {});
        if (!headers.has("Authorization")) headers.set("Authorization", "Bearer " + token);
        next.headers = headers;
      }
    }
    return originalFetch(input, next);
  };

  function openSignIn() {
    if (window.Clerk) window.Clerk.openSignIn();
  }

  // Signing in reloads the page (and Google sign-in leaves the site and comes
  // back), which used to wipe the birth form. The typed details wait in this
  // tab's sessionStorage until the page reopens signed in, then the landing
  // page puts them back and carries on with Generate.
  const PENDING_FORM_KEY = "astromedica_pending_birth_form";
  const PENDING_MAX_AGE_MS = 30 * 60 * 1000;

  function savePendingForm(form) {
    const fields = {};
    form.querySelectorAll("input[id], select[id], textarea[id]").forEach(function (el) {
      fields[el.id] = el.value;
    });
    const geo = document.getElementById("geo-status");
    try {
      sessionStorage.setItem(PENDING_FORM_KEY, JSON.stringify({
        savedAt: Date.now(), fields: fields, geoStatus: geo ? geo.innerHTML : "",
      }));
    } catch (_err) { /* storage blocked: the user re-enters the details, as before */ }
  }

  // Returns the saved details once (and forgets them), or null.
  window.astromedicaTakePendingBirthForm = function () {
    let saved = null;
    try {
      saved = JSON.parse(sessionStorage.getItem(PENDING_FORM_KEY) || "null");
      sessionStorage.removeItem(PENDING_FORM_KEY);
    } catch (_err) { return null; }
    if (!saved || !saved.fields || Date.now() - (saved.savedAt || 0) > PENDING_MAX_AGE_MS) return null;
    return saved;
  };

  window.astromedicaIsSignedIn = function () {
    return !signInRequired || Boolean(window.Clerk && window.Clerk.user);
  };

  document.addEventListener("submit", function (event) {
    const form = event.target;
    if (!form || !form.classList || !form.classList.contains("birth-form")) return;
    if (!signInRequired) return;
    if (!window.Clerk || !window.Clerk.user) {
      event.preventDefault();
      event.stopImmediatePropagation();
      savePendingForm(form);
      openSignIn();
    }
  }, true);

  document.addEventListener("click", function (event) {
    const button = event.target.closest(".mobile-nav-toggle");
    if (!button) return;
    const links = document.querySelector(".nav-links");
    if (links) links.classList.toggle("open");
  });

  window.astromedicaSignOut = async function () {
    if (window.Clerk) await window.Clerk.signOut();
    window.location.href = "/";
  };

  // Resolves once requests can carry the login token (or right away when login
  // is off), so pages wait for it before asking the server about the account.
  let markReady;
  window.astromedicaAuthReady = new Promise(function (resolve) { markReady = resolve; });

  loadConfig().then(async function (config) {
    const slot = document.getElementById("account-slot");
    // No key = login switched off on the server (AUTH_ENABLED); show nothing.
    if (!config.clerkPublishableKey) { markReady(); return; }
    signInRequired = true;
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/@clerk/clerk-js@5/dist/clerk.browser.js";
    script.async = true;
    script.crossOrigin = "anonymous";
    script.setAttribute("data-clerk-publishable-key", config.clerkPublishableKey);
    script.addEventListener("load", async function () {
      await window.Clerk.load();
      markReady();
      if (slot && window.Clerk.user) {
        window.Clerk.mountUserButton(slot);
      } else if (slot) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = "Sign in";
        button.addEventListener("click", openSignIn);
        slot.replaceChildren(button);
        window.Clerk.addListener(function () {
          if (window.Clerk.user) {
            // Reload so the page shows this account's chart (or the form).
            window.location.reload();
          }
        });
      }
    });
    script.addEventListener("error", function () { markReady(); });
    document.head.appendChild(script);
  }).catch(function () { markReady(); });
})();
