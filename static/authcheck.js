/*
 * authcheck.js — 前台「预览 + 登录门禁」探针（整合副本专用）。
 *
 * 行为（2026-11-26 用户定稿）：
 *   - 打开页面默认「预览模式」：可浏览模板、hover 播放封面、点开灯箱看样片、
 *     搜索/筛选分类。不强制跳转登录页。
 *   - 任何「编辑/操作」动作（选模板、换步骤、上传、改设置、生成、下载）在未登录时
 *     被拦截，弹出右上角 Magento 风格 SIGN IN 下拉框提示登录。
 *   - 登录成功把邮箱写入 localStorage（下次预填）；会话 Cookie 服务端 30 天持久化，
 *     下次打开无需再次登录。
 *   - 已登录：右上角显示 MY ACCOUNT（进账号页）+ SIGN OUT（深色按钮），
 *     不显示邮箱/积分（2026-11-26 用户定稿），不拦截任何操作。
 *   - 忘记密码：点击先弹英文提示（到 JP 官网重置），确认后再跳转。
 *
 * 红线：本脚本只调用 /api/me、/api/login、/api/logout，绝不在前端暴露任何口令/密钥。
 *      客户密码只经 POST /api/login 传给 Magento，前端不留痕。
 */
(function () {
  var HOST = location.origin;
  var state = { loggedIn: false, session: null, quota: null };

  /* ---------- 预览白名单：这些元素的交互在「未登录」时也放行 ---------- */
  var PREVIEW_OK = [
    "#tplSearch",           // 模板搜索框
    "#tplChips",            // 分类 chips
    ".tpl-thumb",           // 封面（hover 播放、点开灯箱）
    "[data-act='preview']", // 模板 ▶ 按钮
    "#tplLightbox",         // 样片灯箱
    "#photoLightbox",       // 上传图灯箱
    "#authGate"             // 右上角登录组件自身
  ];
  // 编辑操作：未登录点击这些即拦截并提示登录。
  var EDIT_SEL = "button, input, select, textarea, a[download], label, .drop, .rail-item, .ratios, .upload-pill, .setup-btn";

  function inPreview(el) {
    if (!el || !el.closest) return false;
    for (var i = 0; i < PREVIEW_OK.length; i++) {
      if (el.closest(PREVIEW_OK[i])) return true;
    }
    return false;
  }

  function attrEscape(s) {
    return String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/"/g, "&quot;")
      .replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  /* ---------- 忘记密码：先弹英文提示，确认后到 JP 官网重置 ---------- */
  var JP_FORGOT_URL = "https://www.janicepaper.com/customer/account/forgotpassword/";
  function forgotPassword(ev) {
    if (ev) ev.preventDefault();
    var ok = window.confirm(
      "Your password is managed on the JanicePaper website.\n\n" +
      "To reset it, you'll be taken to JanicePaper.com, where you can recover " +
      "your password using your registered email address.\n\n" +
      "Continue to JanicePaper now?"
    );
    if (ok) window.open(JP_FORGOT_URL, "_blank", "noopener");
  }
  window.__jpForgotPassword = forgotPassword;

  /* ---------- 右上角组件 ---------- */
  function ensureAuthUI() {
    var existing = document.getElementById("authGate");
    if (existing) return existing;
    var header = document.querySelector("header.top") || document.body;
    var box = document.createElement("div");
    box.id = "authGate";
    box.style.cssText = "position:relative;margin-left:auto;display:flex;align-items:center;gap:10px;" +
      "font:500 13px Inter,system-ui,sans-serif;color:var(--ink,#221f1b)";
    header.appendChild(box);
    return box;
  }

  function renderSignedOut() {
    var box = ensureAuthUI();
    var remembered = "";
    try { remembered = localStorage.getItem("jp_studio_email") || ""; } catch (e) {}
    box.innerHTML =
      '<button type="button" id="siBtn" aria-label="Sign in" style="' +
        'display:flex;align-items:center;gap:7px;padding:8px 14px;border:1px solid var(--line-strong,#ded7ca);' +
        'border-radius:3px;background:transparent;cursor:pointer;font:600 12px Inter,system-ui,sans-serif;' +
        'letter-spacing:.08em;color:var(--ink,#221f1b)">' +
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="4" y="10" width="16" height="11" rx="2"/>' +
        '<path d="M8 10V7a4 4 0 0 1 8 0v3"/></svg>SIGN IN</button>' +
      '<div id="siDrop" hidden style="' +
        'position:absolute;top:calc(100% + 8px);right:0;z-index:60;width:300px;padding:18px;' +
        'background:var(--surface,#fff);border:1px solid var(--line-strong,#ded7ca);border-radius:4px;' +
        'box-shadow:0 8px 26px rgba(34,31,27,.13)">' +
        '<div id="siHint" hidden style="margin:0 0 12px;padding:8px 10px;border-radius:3px;' +
        'background:#f4efe6;color:#6b5b3e;font:500 12px Inter,system-ui,sans-serif;line-height:1.5">' +
        'Please sign in to continue editing.</div>' +
        '<label style="display:block;font:600 10px Inter,system-ui,sans-serif;letter-spacing:.12em;' +
        'text-transform:uppercase;color:var(--ink-2,#6b655c);margin-bottom:6px">Email</label>' +
        '<input id="siEmail" type="email" autocomplete="username" placeholder="you@example.com" value="' +
        attrEscape(remembered) + '" style="width:100%;box-sizing:border-box;padding:10px 11px;margin-bottom:12px;' +
        'border:1px solid var(--line-strong,#ded7ca);border-radius:3px;background:var(--surface,#fff);' +
        'color:var(--ink,#221f1b);font:14px Inter,system-ui,sans-serif;outline:none">' +
        '<label style="display:block;font:600 10px Inter,system-ui,sans-serif;letter-spacing:.12em;' +
        'text-transform:uppercase;color:var(--ink-2,#6b655c);margin-bottom:6px">Password</label>' +
        '<input id="siPw" type="password" autocomplete="current-password" style="width:100%;box-sizing:border-box;' +
        'padding:10px 11px;margin-bottom:12px;border:1px solid var(--line-strong,#ded7ca);border-radius:3px;' +
        'background:var(--surface,#fff);color:var(--ink,#221f1b);font:14px Inter,system-ui,sans-serif;outline:none">' +
        '<button type="button" id="siSubmit" style="width:100%;padding:11px;border:1px solid var(--ink,#221f1b);' +
        'border-radius:3px;background:var(--ink,#221f1b);color:#f7f4ee;cursor:pointer;' +
        'font:600 13px Inter,system-ui,sans-serif;letter-spacing:.06em">LOGIN</button>' +
        '<div id="siErr" hidden style="margin-top:10px;font:500 12px Inter,system-ui,sans-serif;color:#a44a3f"></div>' +
        '<div style="margin-top:14px;text-align:center">' +
        '<a href="' + JP_FORGOT_URL + '" id="siForgot" ' +
        'style="font:500 11px Inter,system-ui,sans-serif;letter-spacing:.05em;color:var(--ink-2,#6b655c)">' +
        'FORGOT YOUR PASSWORD?</a></div>' +
      '</div>';
    bindSignIn();
  }

  function renderSignedIn(session, quota) {
    var box = ensureAuthUI();
    var isAdmin = session && session.is_admin;
    // 2026-11-26 用户定稿：不再显示邮箱/积分，改为 MY ACCOUNT 按钮进账号页。
    box.innerHTML =
      '<a href="/account.html" id="acctBtn" aria-label="My account" style="' +
        'display:flex;align-items:center;gap:7px;padding:8px 14px;border:1px solid var(--line-strong,#ded7ca);' +
        'border-radius:3px;background:transparent;cursor:pointer;text-decoration:none;' +
        'font:600 12px Inter,system-ui,sans-serif;letter-spacing:.08em;color:var(--ink,#221f1b)">' +
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>' +
        '<circle cx="12" cy="7" r="4"/></svg>MY ACCOUNT</a>' +
      '<button type="button" id="soBtn" aria-label="Sign out" style="' +
        'display:flex;align-items:center;gap:7px;padding:8px 13px;border:1px solid var(--ink,#221f1b);' +
        'border-radius:3px;background:#2e2a25;color:#f7f4ee;cursor:pointer;' +
        'font:600 12px Inter,system-ui,sans-serif;letter-spacing:.08em">' +
        '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 17l5-5-5-5"/>' +
        '<path d="M15 12H3"/><path d="M14 3h5a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-5"/></svg>SIGN OUT</button>';
    document.getElementById("soBtn").addEventListener("click", function () {
      fetch(HOST + "/api/logout", { method: "POST", credentials: "same-origin" })
        .finally(function () { location.href = "/"; }); // 退出后回到预览模式
    });
  }

  function bindSignIn() {
    var btn = document.getElementById("siBtn");
    var drop = document.getElementById("siDrop");
    var hint = document.getElementById("siHint");
    var email = document.getElementById("siEmail");
    var pw = document.getElementById("siPw");
    var err = document.getElementById("siErr");
    btn.addEventListener("click", function (e) {
      e.preventDefault();
      drop.hidden = !drop.hidden;
      if (!drop.hidden) { err.hidden = true; (email.value ? pw : email).focus(); }
    });
    document.addEventListener("click", function (e) {
      if (drop.hidden) return;
      var gate = document.getElementById("authGate");
      if (gate && !gate.contains(e.target)) drop.hidden = true;
    });
    function submit() {
      var em = email.value.trim();
      var pass = pw.value;
      err.hidden = true;
      if (!em || !pass) {
        err.textContent = "Please enter your email and password."; err.hidden = false; return;
      }
      var sub = document.getElementById("siSubmit");
      sub.disabled = true; sub.textContent = "SIGNING IN…";
      fetch(HOST + "/api/login", {
        method: "POST", credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: em, password: pass })
      }).then(function (r) {
        if (r.ok) {
          try { localStorage.setItem("jp_studio_email", em); } catch (e2) {}
          location.reload();
          return null;
        }
        return r.json().catch(function () { return {}; }).then(function (j) {
          var msg = j.error || "";
          var text = /停用|disabled/i.test(msg) ? "This account has been disabled."
            : /暂时不可用|unavailable/i.test(msg) ? "Sign-in service is temporarily unavailable. Please try again."
            : "Invalid email or password.";
          err.textContent = text; err.hidden = false;
          sub.disabled = false; sub.textContent = "LOGIN";
        });
      }).catch(function () {
        err.textContent = "Network error. Please try again."; err.hidden = false;
        sub.disabled = false; sub.textContent = "LOGIN";
      });
    }
    document.getElementById("siSubmit").addEventListener("click", submit);
    var forgot = document.getElementById("siForgot");
    if (forgot) forgot.addEventListener("click", forgotPassword);
    [email, pw].forEach(function (el) {
      el.addEventListener("keydown", function (e) { if (e.key === "Enter") submit(); });
    });
    // 被 Edit 拦截时由此函数弹出提示
    window.__jpPromptLogin = function (message) {
      renderSignedOut();
      var d = document.getElementById("siDrop");
      var h = document.getElementById("siHint");
      if (d) { d.hidden = false; if (h && message) { h.textContent = message; h.hidden = false; } }
      var pe = document.getElementById("siEmail");
      if (pe) pe.focus();
    };
  }

  /* ---------- 预览门禁：捕获阶段拦截编辑操作 ---------- */
  var gateInstalled = false;
  function installGate() {
    if (gateInstalled) return;
    gateInstalled = true;
    function onEdit(e) {
      if (state.loggedIn) return;                  // 已登录 → 全部放行
      if (e.type === "keydown" && e.key !== "Enter" && e.key !== " ") return;
      var el = e.target;
      if (!el) return;
      if (el.nodeType === 3) el = el.parentNode;   // 文本节点
      if (inPreview(el)) return;                   // 预览白名单 → 放行
      if (!el.closest || !el.closest(EDIT_SEL)) return; // 非交互元素 → 放行
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      if (typeof window.__jpPromptLogin === "function") {
        window.__jpPromptLogin("Please sign in to create your video.");
      }
    }
    document.addEventListener("click", onEdit, true);
    document.addEventListener("keydown", onEdit, true);
  }

  /* ---------- 身份探测 ---------- */
  function applyState(loggedIn, session, quota) {
    state.loggedIn = loggedIn; state.session = session || null; state.quota = quota || null;
    document.documentElement.setAttribute("data-auth", loggedIn ? "in" : "out");
    if (loggedIn) {
      renderSignedIn(session, quota);
      // boot 在探针前已按默认预览态启动（不建会话）；探测到已登录后补建，
      // 保证上传/生成可用。未登录时绝不调用。
      try {
        if (window.__studio && typeof window.__studio.ensureSession === "function") {
          window.__studio.ensureSession().catch(function () {});
        }
      } catch (e) {}
    } else renderSignedOut();
    installGate();
  }

  function check() {
    // 登录页/记录页本身不做门禁（记录页有自己的访问口令逻辑）
    if (location.pathname === "/login.html" || location.pathname === "/records.html") return;
    fetch(HOST + "/api/me", { method: "GET", headers: { "Accept": "application/json" },
      credentials: "same-origin" })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (res) {
        if (!res.ok || (res.j && res.j.error)) { applyState(false); return; }
        applyState(true, res.j.session, res.j.quota);
      })
      .catch(function () { applyState(false); });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", check);
  else check();
})();
