/* 请柬推广素材工作台 - 前端逻辑（4 步版：上传→AI 处理元素素材→视频生成→合成导出） */
"use strict";

const S = {
  projects: [], project: null, step: 1,
  config: {}, bgmTracks: [],
  segOrder: [],          // 步骤4: 视频段顺序 [{path, name, kind, dur}]
  segSel: null,          // 步骤4: 勾选的段 key 集合
  segSelPid: null,       // 段选择所属项目（换项目时重置）
  cmSel: [],             // （保留字段，向后兼容；基础运镜已固定五镜序列，不再勾选）
  cmPreviewDone: false,  // 运镜: 是否已预览
  elMode: null,          // 步骤2: 当前处理模式（fidelity / restage）
  elEdited: false,       // 步骤2: 指令词是否被手动编辑过（编辑后不再自动覆盖）
  elPid: null,           // 步骤2: 指令词所属项目（换项目时重置编辑标记）
  sessionIds: [],        // 本次会话（=本次打开页面）新建的项目，列表只显示这些
  showAll: false,        // 手动「载入历史项目」后为 true
  shotLists: {},         // 拍摄清单内存缓存（随会话结束丢弃）
};

/* 官方指令词（逐字固化；启动时会以 /api/prompts 覆盖，此处为离线兜底）
   改动口径：这两段同学生产口径的一部分，请勿随手改措辞。 */
const FALLBACK_PROMPTS = {
  video_handheld:
    "以参考图片、首帧和尾帧作为唯一视觉依据，生成一段真实自然的婚礼请柬产品展示视频。" +
    "请先识别图片中的所有独立纸品组件和卡片内容，并按照常规请柬产品展示顺序依次拿起展示，再放回原位。" +
    "整个视频必须保持同一桌面、同一背景、同一道具、同一光线、同一机位和同一双手，" +
    "所有产品在整个视频中的颜色、形状、尺寸比例、结构、边缘、纸张质感、文字内容、字体、排版、花纹、插画、" +
    "二维码及所有工艺细节必须100%保持一致，不得发生任何变化。\n" +
    "视频开始时，所有产品保持与首帧一致的桌面摆放状态，双手自然从画面下方进入，准备开始展示。" +
    "卡片的拿起顺序按照常规产品展示顺序执行：优先展示主请柬/Invitation，然后展示RSVP卡，" +
    "再展示Details卡 / Reception卡 / Information卡，然后展示Menu卡，" +
    "如有其他小卡则继续按辅助卡顺序展示；如需要展示信封、外壳、套壳，则放在最后展示，或保持作为背景辅助元素不动。\n" +
    "每次只拿起一张卡片，其余卡片保持留在桌面原位不动。每张卡片都按照统一动作进行展示：" +
    "双手轻触卡片左右下方边缘，缓慢将卡片从桌面拿起，移动到画面中央进行正面展示，保持卡片始终正对镜头，" +
    "仅允许非常轻微的自然手部微动作以及约3°–5°的小幅度左右倾斜，可轻微向镜头靠近约5%，" +
    "突出纸张质感和印刷/工艺细节。每张卡片在画面中央稳定展示约0.5–1秒后，再沿原路径缓慢放回桌面，" +
    "尽量恢复原来的摆放位置与角度，然后再拿起下一张卡片重复相同动作。\n" +
    "所有卡片都展示完成后，双手完成最后一张卡片的放回动作，并使整套产品恢复到尾帧所对应的整齐桌面摆放状态。" +
    "结尾必须有明显停留：最后一张卡片放回后，镜头保持静止，整套产品完整留在画面中稳定停留约1–1.5秒，" +
    "作为最终收尾展示。此时双手可以轻触最后一张卡片边缘，或自然缓慢退出画面下方，但不得破坏整体构图与产品摆放。\n" +
    "镜头保持固定或仅有极轻微自然推进，禁止快速运镜、绕拍、大幅旋转和突然视角变化。" +
    "动作节奏缓慢、自然、稳定、连贯。禁止：产品变形、结构错误、文字乱码、花纹变化、二维码变化、背景变化、" +
    "手部异常、手指增减、卡片凭空消失、额外卡片出现、道具变化、明显AI感、3D感、塑料感、动作跳变。",
  storyboard_handheld:
    "请根据我上传的产品图，先判断产品的真实结构，再按结构生成手持展示分镜图：\n" +
    "第一步，区分「独立小卡」和「与外壳/载体一体的内容」：只有真实独立存在的小卡才单独出镜；" +
    "凡是印在外壳、内页或折页上的内容（如 Reception 栏、RSVP 栏、Details 栏），没有独立单卡的，" +
    "严禁拆分成独立卡片，也严禁凭空编造画面中不存在的小卡，必须以其载体（信封外壳、三折内页、折卡等）" +
    "整体作为展示主体，一张分镜图只展示一个主体，不要拼图。\n" +
    "第二步，以载体整体展示并突出工艺细节：外壳类（信封+内衬+封缄贴）做整体手持展示，" +
    "通过轻微角度变化和自然光突出压凹纹理、烫金封缄、内衬花纹、纸张毛边等工艺细节；" +
    "折页/内页类做整体手持展示，完整呈现其展开形态与栏目排版，突出烫金文字、花纹边框与印刷质感。\n" +
    "第三步，锁定真实比例：产品在画面中的大小、与手部的比例，必须与上传产品图中的真实尺寸关系一致，" +
    "不得把纸品放大占满画面，不得改变产品的长宽比例、厚度和折叠结构。\n" +
    "每一张分镜图都要表现为“真人双手手持产品进行展示”的镜头效果，适合作为后续视频生成的手持展示分镜。" +
    "双手自然拿住产品边缘，避免遮挡主要文字、图案和工艺细节，产品正面完整清晰可见，居中展示，画面稳定，构图简洁高级。\n" +
    "严格保持产品原始设计100%一致，不得修改或重绘产品内容，包括产品的颜色、形状、尺寸比例、边缘造型、" +
    "纸张质感、文字内容、字体、排版、花纹、插画、二维码及工艺细节。不得新增错误文字，不得改变产品比例和结构。\n" +
    "背景需要延续并还原上传产品图的整体拍摄环境和布景风格，保持与原图一致的白色/浅色背景基调、花材、托盘、" +
    "丝带、装饰物、光线氛围和整体色调。允许背景做浅景深虚化处理，但不能更换为全新场景，也不要加入无关装饰物。" +
    "整体效果要像从原场景中自然延伸出来的手持实拍镜头。\n" +
    "画面风格要求为高质感写实摄影风格，真实自然光，纸张材质真实，手部结构自然，五指正常无畸形，" +
    "适合婚礼请柬产品的社交媒体推广视频分镜。每张输出为单独的竖版分镜图，建议9:16比例。\n" +
    "禁止：把非独立卡拆成单卡、凭空编造不存在的小卡、产品比例失真、产品放大占满画面、拼图、多个主体混在一图、" +
    "背景大幅更换、产品变形、文字乱码、错误图案、错误二维码、手指异常、明显AI感、3D/CG塑料质感、过度夸张动作。",
  /* ---- 步骤2 · AI 处理元素素材（两种模式 + 两种输出粒度） ---- */
  element_fidelity:
    "请严格依据我上传的产品实拍图，先识别画面中所有真实存在的独立纸品元素" +
    "（例如主请柬/Invitation、RSVP卡、Details卡、Reception卡、Information卡、Menu卡、" +
    "信封外壳、内衬、封缄贴、缎带及其他辅助小卡），再逐个输出修复后的元素素材图。\n" +
    "每个元素单独输出一张图，一张图只呈现一个元素，不要拼图，不要把多个元素混在同一画面中。\n" +
    "修复范围仅限以下项目：清除画面中的手部、绳带杂物、桌面污渍、反光与生硬阴影、平台水印与贴纸；" +
    "修补被裁切或破损的纸张边缘；提升分辨率与细节清晰度；校正倾斜与透视变形；" +
    "统一光位方向与白平衡，使整组素材风格一致。\n" +
    "严禁改变产品的任何设计信息：颜色、形状、尺寸比例、结构、折叠方式、边缘造型、纸张材质与克重感、" +
    "文字内容、字体、字号、排版位置、花纹、插画、二维码，以及烫金、压凹、Letterpress 等所有工艺细节，" +
    "必须与上传原图完全一致。严禁重绘产品，严禁新增或修改任何文字，严禁编造原图中不存在的元素。\n" +
    "输出为可用于后续视频生成的正面清晰素材：元素正面完整入画、边缘不被裁切、构图居中、背景干净统一，" +
    "保持与原图一致的纸张质感与真实自然光。\n" +
    "画面风格：高质感写实商业摄影，真实纸张纹理，细腻的烫金与压凹质感，轻微景深。\n" +
    "禁止：改变产品设计、文字乱码、字体替换、排版位移、花纹变化、二维码变化、尺寸比例失真、产品变形、" +
    "拉伸、融化、拼接痕、明显重绘痕迹、AI感、3D感、塑料感。",
  element_restage:
    "请参考我上传的产品实拍图，先识别其中的独立纸品元素、各自的真实尺寸比例与工艺特征，" +
    "然后在严格保持每个元素自身设计100%不变的前提下，重新搭配元素组合并重新构建拍摄场景，" +
    "生成新的产品参考素材图。\n" +
    "允许改变的部分：元素之间的摆放组合、层叠与开合关系、画面构图与机位角度、桌面与背景材质、" +
    "花材与装饰道具、光线方向与氛围色调、整体色调风格。\n" +
    "严禁改变的部分：产品本身的颜色、形状、尺寸比例、结构、折叠方式、边缘造型、纸张材质、文字内容、" +
    "字体、排版、花纹、插画、二维码，以及烫金、压凹、Letterpress 等工艺细节，必须与上传原图完全一致。" +
    "严禁新增原图中不存在的产品类型，严禁编造不存在的卡片或文字，严禁把不同元素的设计相互混用。\n" +
    "新场景必须是真实可落地的拍摄场景：真实桌面材质、真实自然光、真实花材与道具；" +
    "各元素之间的尺寸比例与真实物理关系正确，光影方向统一，投影自然合理，元素实实在在落在桌面上。\n" +
    "画面风格：高质感写实商业摄影，真实纸张质感，细腻的烫金与压凹细节，轻微景深，高级婚礼文具风格。\n" +
    "禁止：改变产品设计、文字乱码、字体替换、排版位移、花纹变化、二维码变化、尺寸比例失真、产品变形、" +
    "元素比例关系错误、光影方向冲突、悬浮不落地、明显AI感、3D/CG塑料质感。",
  element_gran_item:
    "\n\n【本次输出要求】逐个元素输出：识别到的每个独立元素各输出一张图，" +
    "一张图只呈现一个元素，不要拼图，不要把多个元素放进同一画面。",
  element_gran_set:
    "\n\n【本次输出要求】整套场景输出：把识别到的全部元素按新的搭配组合呈现在同一个场景里，" +
    "输出一张完整的整图参考素材，要像一次真实拍摄的成品照，不要拼图式罗列、不要分格排版。"
};
const P = { ...FALLBACK_PROMPTS };

/* ---------------- 基础工具 ---------------- */
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

/* 本工作台不做任何跨会话记忆：每次打开/刷新都是干净起点。
   这里只负责清掉历史版本遗留的 localStorage 记忆，避免旧进度被意外带出。 */
(function purgeLegacyMemory() {
  try {
    const doomed = [];
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && (k === "wb.curProject" || k === "wb.curStep" || k.indexOf("wb.shotlist.") === 0)) doomed.push(k);
    }
    doomed.forEach(k => localStorage.removeItem(k));
  } catch (e) { /* 无 localStorage 时忽略 */ }
})();

async function api(path, body) {
  const opt = body ? {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  } : {};
  const r = await fetch(path, opt);
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}

async function uploadFile(file, kind) {
  const pid = S.project.id;
  const url = `/api/upload?project=${encodeURIComponent(pid)}&name=${encodeURIComponent(file.name)}&kind=${kind || "upload"}`;
  const r = await fetch(url, { method: "POST", body: file });
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  S.project = j.project;
  return j.entry;
}

function toast(msg, ms = 2600) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.hidden = true, ms);
}

function pollJob(jid, box, onDone) {
  const timer = setInterval(async () => {
    try {
      const { job } = await api("/api/job?id=" + jid);
      if (!job) { clearInterval(timer); return; }
      if (job.status === "running") {
        box.innerHTML = `<div class="job-msg">${job.msg || "Processing"} ${job.progress}%</div>
          <div class="progress"><div style="width:${job.progress}%"></div></div>`;
      } else {
        clearInterval(timer);
        if (job.status === "error") {
          box.innerHTML = `<div class="job-error">✕ Failed: ${job.error}</div>`;
        } else if (onDone) onDone(job.result);
      }
    } catch (e) { clearInterval(timer); box.innerHTML = `<div class="job-error">✕ ${e.message}</div>`; }
  }, 1500);
}

const media = (p) => p;  // 服务端已返回 /media/... 相对路径
function fmtDur(s) { return s ? (Math.round(s * 10) / 10 + "s") : ""; }

/* ---------------- 步骤切换（4 步） ---------------- */
function goto(step) {
  step = Math.min(4, Math.max(1, +step || 1));
  if (!S.project && step > 1) { toast("Create a project first (\"New Project\" at the top right)"); step = 1; }
  S.step = step;
  $$(".step").forEach(el => {
    const n = +el.dataset.step;
    el.classList.toggle("active", n === step);
    el.classList.toggle("done", n < step);
  });
  $$(".panel").forEach(el =>
    el.classList.toggle("show", +el.dataset.panel === step));
  renderStep(step);
}
$$(".step").forEach(el => el.addEventListener("click", () => goto(+el.dataset.step)));
$$("[data-goto]").forEach(b => b.addEventListener("click", () => goto(+b.dataset.goto)));

/* ---------------- 项目 ---------------- */
/* 列表只显示「本次会话新建」的项目；历史项目默认不出现，可用「载入历史项目」临时查看。 */
async function refreshProjects(selectId) {
  const { projects } = await api("/api/projects");
  S.projects = projects;
  const visible = S.showAll ? projects : projects.filter(p => S.sessionIds.indexOf(p.id) >= 0);
  const sel = $("#projectSelect");
  sel.innerHTML = visible.map(p =>
    `<option value="${p.id}">${p.name}</option>`).join("") ||
    `<option value="">(no project in this session — create one)</option>`;
  if (selectId && visible.some(p => p.id === selectId)) sel.value = selectId;
  S.project = visible.find(p => p.id === sel.value) || null;
  renderAll();
}

$("#projectSelect").addEventListener("change", async () => {
  const pid = $("#projectSelect").value;
  if (!pid) { S.project = null; renderAll(); return; }
  const { project } = await api("/api/project?id=" + pid);
  S.project = project;
  renderAll();
});

async function newProject(presetName) {
  const name = presetName !== undefined ? presetName
    : prompt("Project name (e.g. Christina & Justin white-metal suite)",
        "New project " + new Date().toLocaleString("zh-CN", { hour12: false }));
  if (name === null) return null;
  const { project } = await api("/api/project", { name: name || "Untitled project" });
  if (S.sessionIds.indexOf(project.id) < 0) S.sessionIds.push(project.id);
  await refreshProjects(project.id);
  toast("New session project created: " + project.name);
  return project;
}
$("#btnNewProject").addEventListener("click", async () => {
  const p = await newProject();
  if (p) goto(1);
});
$("#btnStartProject").addEventListener("click", async () => {
  const p = await newProject();
  if (p) goto(1);
});
$("#btnLoadHistory").addEventListener("click", async () => {
  S.showAll = true;
  await refreshProjects();
  toast("Past project loaded temporarily (visible this session only; refreshing restores a clean start)", 3600);
});

/* 上传时若还没有项目：自动新建一个本次会话项目，避免「拖进来没反应」 */
async function ensureProject() {
  if (S.project) return true;
  await newProject("Session project " + new Date().toLocaleString("zh-CN", { hour12: false }));
  return !!S.project;
}

/* ---------------- 设置 ---------------- */
let VID_MODELS = [], IMG_MODELS = [];
const openSettings = () => { $("#settingsModal").hidden = false; };
const closeSettings = () => { $("#settingsModal").hidden = true; };
window.__openSettings = openSettings;   // 供其他分支调用
window.__closeSettings = closeSettings;

$("#btnSettings").addEventListener("click", async () => {
  const { config } = await api("/api/config");
  S.config = config;
  VID_MODELS = config.video_models || [];
  IMG_MODELS = config.image_models || [];
  const v = config.video_api || {};
  // 火山方舟
  $("#cfgVidEp").value = v.endpoint || "";
  $("#cfgVidEp").placeholder = (v.effective_endpoint || "https://ark.cn-beijing.volces.com/api/v3") + " (leave blank for default)";
  $("#cfgVidKey").value = v.api_key || "";
  $("#cfgVidKey").placeholder = v.api_key_set ? "Configured (enter a new value to update)" : "Paste your Volcengine Ark long-lived API key";
  $("#cfgVidModel").innerHTML = VID_MODELS.map(m =>
    `<option value="${m.id}">${m.name}</option>`).join("");
  const mk = v.model || (VID_MODELS[0] || {}).id || "";
  if (mk && !VID_MODELS.some(m => m.id === mk))
    $("#cfgVidModel").insertAdjacentHTML("afterbegin", `<option value="${mk}">${mk} (custom)</option>`);
  $("#cfgVidModel").value = mk;
  $("#cfgVidRes").value = v.resolution || "720p";
  $("#cfgVidRatio").value = v.ratio || "9:16";
  $("#cfgVidAudio").checked = !!v.generate_audio;
  $("#cfgVidWm").checked = !!v.watermark;
  $("#vidTestOut").innerHTML = v.api_key_set
    ? `Current base: <code>${v.effective_endpoint}</code> · model <code>${v.effective_model}</code>`
    : "Not configured — enter an API key, then click \"Test connection\"";
  // 分镜图 · Seedream
  const sb = config.storyboard_api || {};
  $("#cfgSbEp").value = sb.endpoint || "";
  $("#cfgSbEp").placeholder = (sb.effective_endpoint || "https://ark.cn-beijing.volces.com/api/v3") + " (leave blank for default)";
  $("#cfgSbKey").value = sb.api_key || "";
  $("#cfgSbKey").placeholder = sb.api_key_set ? "Configured (enter a new value to update)" : "Can share the same ARK_API_KEY as video generation";
  $("#cfgSbModel").innerHTML = IMG_MODELS.map(m =>
    `<option value="${m.id}">${m.name}</option>`).join("");
  const smk = sb.model || (IMG_MODELS[0] || {}).id || "";
  if (smk && !IMG_MODELS.some(m => m.id === smk))
    $("#cfgSbModel").insertAdjacentHTML("afterbegin", `<option value="${smk}">${smk} (custom)</option>`);
  $("#cfgSbModel").value = smk;
  $("#cfgSbSize").value = sb.size || "4K";
  $("#cfgSbMax").value = sb.max_images || 3;
  $("#cfgSbFmt").value = sb.output_format || "jpeg";
  $("#cfgSbWm").checked = !!sb.watermark;
  $("#cfgSbOpt").checked = sb.optimize_prompt === "standard";
  $("#sbTestOut").innerHTML = sb.api_key_set
    ? `Current base: <code>${sb.effective_endpoint}</code> · model <code>${sb.effective_model}</code>`
    : "Not configured — enter an API key, then click \"Test connection\"";
  openSettings();
});
/* 关闭：× 按钮 / 取消 / 点击遮罩空白处 / Esc 键 */
$("#btnCfgClose").addEventListener("click", closeSettings);
$("#btnCfgCancel").addEventListener("click", closeSettings);
$("#settingsModal").addEventListener("mousedown", (e) => {
  if (e.target === $("#settingsModal")) closeSettings();   // 只有点到遮罩本身才关，点弹窗内部不关
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("#settingsModal").hidden) closeSettings();
});
$("#btnCfgSave").addEventListener("click", async () => {
  const body = { video_api: {}, storyboard_api: {} };
  body.video_api = {
    provider: "volcengine_ark",
    endpoint: $("#cfgVidEp").value.trim(), api_key: $("#cfgVidKey").value.trim(),
    model: $("#cfgVidModel").value.trim(),
    resolution: $("#cfgVidRes").value, ratio: $("#cfgVidRatio").value,
    generate_audio: $("#cfgVidAudio").checked, watermark: $("#cfgVidWm").checked,
    camera_fixed: (S.config.video_api || {}).camera_fixed || false,
  };
  body.storyboard_api = {
    provider: "volcengine_ark",
    endpoint: $("#cfgSbEp").value.trim(), api_key: $("#cfgSbKey").value.trim(),
    model: $("#cfgSbModel").value.trim(),
    size: $("#cfgSbSize").value, output_format: $("#cfgSbFmt").value,
    max_images: +$("#cfgSbMax").value || 3, sequential: "auto",
    watermark: $("#cfgSbWm").checked,
    optimize_prompt: $("#cfgSbOpt").checked ? "standard" : "",
  };
  await api("/api/config", body);
  closeSettings();
  toast("Settings saved");
});
$("#btnVidTest").addEventListener("click", async () => {
  const out = $("#vidTestOut");
  out.textContent = "Testing…";
  try {
    const r = await api("/api/videoapi/test", {
      api_key: $("#cfgVidKey").value.trim(), endpoint: $("#cfgVidEp").value.trim(),
    });
    const n = (r.recent_tasks || []).length;
    out.innerHTML = `<span style="color:#1d6b38">✓ Connected</span> · base <code>${r.base}</code> · model <code>${r.model}</code>` +
      (n ? ` · ${n} recent brief(s) available` : "");
  } catch (e) {
    out.innerHTML = `<span style="color:#8a2020">✕ ${e.message}</span>`;
  }
});
$("#btnSbTest").addEventListener("click", async () => {
  const out = $("#sbTestOut");
  out.textContent = "Checking… (read-only model list, no cost)";
  const mark = (c) => !c ? "" :
    c.found === false
      ? `<span style="color:#8a2020">✕ ${c.id} not enabled</span>`
      : `<span style="color:${c.warning ? "#b7791f" : "#1d6b38"}">${c.warning ? "⚠" : "✓"} ${c.id}（${c.status}）</span>`;
  try {
    const r = await api("/api/imageapi/test", {
      api_key: $("#cfgSbKey").value.trim(), endpoint: $("#cfgSbEp").value.trim(),
      model: $("#cfgSbModel").value.trim(),
    });
    out.innerHTML =
      `${mark(r.image_model_check)}<br>${mark(r.video_model_check)}<br>` +
      `<span class="hint">base <code>${r.base}</code> · available quality ${(r.sizes || []).join("/")}` +
      `${r.recent_tasks !== undefined ? ` · ${r.recent_tasks} recent video task(s)` : ""}</span>` +
      (r.detail ? `<br><span class="hint">${r.detail}</span>` : "");
  } catch (e) {
    out.innerHTML = `<span style="color:#8a2020">✕ ${e.message}</span>`;
  }
});

/* ---------------- 渲染调度 ---------------- */
function renderAll() { for (let i = 1; i <= 4; i++) renderStep(i); }
function renderStep(step) {
  if (step === 1) renderUploads();      // 含「无项目」空状态
  if (!S.project) return;
  if (step === 2) renderElementStep();
  if (step === 3) renderVideoStep();
  if (step === 4) renderCompose();
}

/* ================= 步骤1 上传 ================= */
function bindDropzone(dzId, fileInputId, kind, after) {
  const dz = $(dzId), fi = $(fileInputId);
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", async e => {
    e.preventDefault(); dz.classList.remove("drag");
    await handleFiles(e.dataTransfer.files, kind, after);
  });
  fi.addEventListener("change", async () => {
    await handleFiles(fi.files, kind, after); fi.value = "";
  });
}
async function handleFiles(files, kind, after) {
  if (!files || !files.length) return;
  if (!S.project) {
    await ensureProject();
    if (!S.project) return;
  }
  let n = 0;
  for (const f of files) {
    if (kind === "upload" && !/^image\//.test(f.type)) { toast("Skipped non-image file: " + f.name); continue; }
    try { await uploadFile(f, kind); n++; }
    catch (e) { toast("Upload failed " + f.name + "：" + e.message); }
  }
  if (n) {
    toast(`${n} file(s) uploaded`);
    renderAll();
    if (after) after();
  }
}
bindDropzone("#dzUpload", "#fileUpload", "upload");
bindDropzone("#dzStoryboard", "#fileStoryboard", "storyboard");

function renderUploads() {
  const none = !S.project;
  $("#noProject").hidden = !none;
  const ups = none ? [] : (S.project.uploads || []);
  const g = $("#uploadGrid");
  if (none) { g.innerHTML = ""; return; }
  if (!ups.length) { g.innerHTML = `<div class="empty" style="grid-column:1/-1">No photos uploaded yet</div>`; return; }
  g.innerHTML = ups.map(e => `
    <div class="thumb">
      <img src="/media/${S.project.id}/uploads/${encodeURIComponent(e.file)}" loading="lazy">
      <div class="cap">${e.file}</div>
    </div>`).join("");
}

/* ================= 步骤2 AI 处理元素素材 ================= */
/* 链路：上传实拍图 → AI 拆解元素 → 按模式/粒度输出参考素材 → 成为下游默认来源。
   两种模式：fidelity 保真优化 / restage 重搭配重场景；两种粒度：item 逐个元素 / set 整套场景。 */

let _elLastMode = null;   // 记住上一次渲染时的模式，用于判断「是否该把粒度回落到模式默认」

function elModes() {
  const m = (S.config || {}).element_modes;
  return (m && m.length) ? m : [
    { id: "fidelity", name: "Fidelity restoration", default_gran: "item",
      desc: "Cleanup and faithful restoration only: remove hands / clutter / watermarks, sharpen, correct geometry, unify lighting. Product design stays 100% unchanged." },
    { id: "restage", name: "AI restage & rescene", default_gran: "set",
      desc: "Element designs stay 100% unchanged; recompose the arrangement and rebuild the shooting scene (new tabletop / florals / lighting / framing)." },
  ];
}
function elGrans() {
  const g = (S.config || {}).element_granularities;
  return (g && g.length) ? g : [
    { id: "item", name: "Per element (one image per element)" },
    { id: "set", name: "Full set (all elements in one image)" },
  ];
}
function elModeMeta(id) {
  const list = elModes();
  return list.find(m => m.id === id) || list[0];
}
function elModeVal() {
  const r = document.querySelector('input[name="elmode"]:checked');
  return r ? r.value : (S.elMode || (S.config || {}).element_default_mode || elModes()[0].id);
}
function elGranVal() {
  const v = $("#elGran") ? $("#elGran").value : "";
  return v || elModeMeta(elModeVal()).default_gran || "item";
}
function elDefaultGran(modeId) { return elModeMeta(modeId).default_gran || "item"; }
function elPromptText(modeId, granId) {
  const body = modeId === "restage" ? P.element_restage : P.element_fidelity;
  const suffix = granId === "set" ? P.element_gran_set : P.element_gran_item;
  return (body || "") + (suffix || "");
}
// 未手动编辑过 → 指令词跟随模式/粒度自动替换；force=true 用于「恢复官方指令」
function syncElPrompt(force) {
  const ta = $("#elPrompt");
  if (!ta) return;
  if (force) S.elEdited = false;
  if (!S.elEdited) ta.value = elPromptText(elModeVal(), elGranVal());
}

function renderElMode() {
  const box = $("#elMode");
  if (!box) return;
  const modes = elModes();
  const def = (S.config || {}).element_default_mode || modes[0].id;
  if (!S.elMode) S.elMode = def;
  if (!modes.some(m => m.id === S.elMode)) S.elMode = def;
  if (box.dataset.ready !== String(modes.length)) {
    box.innerHTML = modes.map(m =>
      `<label><input type="radio" name="elmode" value="${m.id}"${m.id === S.elMode ? " checked" : ""}>
        <b>${m.name}</b><span class="hint" style="margin:0">${m.desc || ""}</span></label>`).join("");
    Array.from(box.querySelectorAll('input[name="elmode"]')).forEach(r =>
      r.addEventListener("change", () => {
        S.elMode = r.value;
        renderElGranOptions();      // 换模式 → 粒度回落到该模式默认（内部处理）
        syncElPrompt(false);
        renderElNotes();
      }));
    box.dataset.ready = String(modes.length);
  } else {
    const r = box.querySelector(`input[value="${S.elMode}"]`);
    if (r) r.checked = true;
  }
}

function renderElGranOptions() {
  const sel = $("#elGran");
  if (!sel) return;
  const grans = elGrans();
  const mode = elModeVal();
  const prev = sel.value;
  sel.innerHTML = grans.map(g => `<option value="${g.id}">${g.name}</option>`).join("");
  if (_elLastMode !== mode) { sel.value = elDefaultGran(mode); _elLastMode = mode; }
  else if (grans.some(g => g.id === prev)) sel.value = prev;
  else sel.value = elDefaultGran(mode);
  if (!sel.dataset.bound) {
    sel.addEventListener("change", () => { syncElPrompt(false); renderElCountOptions(); renderElNotes(); });
    sel.dataset.bound = "1";
  }
  renderElCountOptions();
}

function elRefCount() {
  const v = $("#elRef") ? $("#elRef").value : "";
  if (v) return 1;
  return Math.min(3, (S.project.uploads || []).length) || 1;
}

function renderElCountOptions() {
  const sel = $("#elCount");
  if (!sel) return;
  const gran = elGranVal();
  const cap = Math.max(1, 15 - elRefCount());      // 方舟：参考图数 + 生成图数 ≤ 15
  const prev = +sel.value || 0;
  const opts = gran === "set" ? [1]
    : Array.from({ length: Math.min(cap, 12) }, (_, i) => i + 1);
  sel.innerHTML = opts.map(n => `<option value="${n}">${n} images</option>`).join("");
  sel.value = gran === "set" ? 1 : (opts.indexOf(prev) >= 0 ? prev : Math.min(6, cap));
}

function renderElRefOptions() {
  const sel = $("#elRef");
  if (!sel) return;
  const cur = sel.value;
  const els = S.project.elements || [];
  const ups = S.project.uploads || [];
  const prc = Object.values(S.project.processed || {});
  let html = `<option value="">(Auto: uses all step 1 uploads, up to 3)</option>`;
  if (els.length) html += `<optgroup label="Element assets (step 2 output · can be refined again)">` +
    els.map(e => `<option value="${e.path}">${e.file}</option>`).join("") + `</optgroup>`;
  if (ups.length) html += `<optgroup label="Step 1 uploads">` +
    ups.map(e => `<option value="${e.path}">${e.file}</option>`).join("") + `</optgroup>`;
  if (prc.length) html += `<optgroup label="Local processing output">` +
    prc.map(e => `<option value="${e.path}">${e.file}</option>`).join("") + `</optgroup>`;
  sel.innerHTML = html;
  if (cur && Array.from(sel.options).some(o => o.value === cur)) sel.value = cur;
  if (!sel.dataset.bound) {
    sel.addEventListener("change", () => { renderElCountOptions(); renderElNotes(); });
    sel.dataset.bound = "1";
  }
  renderElCountOptions();
}

function renderElSizeOptions() {
  const sel = $("#elSize");
  if (!sel) return;
  const meta = ((S.config || {}).storyboard_api || {}).meta || {};
  const arr = meta.sizes || ["2K", "3K", "4K"];
  const cur = sel.value;
  sel.innerHTML = arr.map(s => `<option value="${s}">${s}</option>`).join("");
  sel.value = arr.indexOf(cur) >= 0 ? cur : (arr[arr.length - 1] || "4K");
  if (!sel.dataset.bound) {
    sel.addEventListener("change", renderElNotes);
    sel.dataset.bound = "1";
  }
}

function renderElArkState() {
  const sb = (S.config || {}).storyboard_api || {};
  const box = $("#elArkState");
  if (!box) return;
  if (sb.api_key_set) {
    const m = sb.meta || {};
    box.className = "banner ok";
    box.innerHTML = `Volcengine Ark ready · model <code>${sb.effective_model || ""}</code> · available quality ${(m.sizes || ["2K", "3K", "4K"]).join("/")}
      · Set constraint: reference images + generated images ≤ 15<br>
      Billed by the number of images actually generated (4K about ¥0.22 each; less at 2K/3K). Output is written to the project's <code>elements/</code> folder and becomes step 3's default reference source.<br>
      <span style="color:#8a2020">Note: AI redrawing is not character-accurate on small type — for text-dense cards, trust the original photo or design source file.</span>`;
  } else {
    box.className = "banner warn";
    box.innerHTML = `ARK_API_KEY is not configured, so AI processing is unavailable. Go to <b>Settings → Image generation API</b> and enter your Volcengine Ark long-lived key (shared with video generation). Until then you can use the local operators below as a fallback.<br>
      This step is <b>optional</b> — you can skip it entirely with "Skip — use original uploads" below.`;
  }
}

function renderElNotes() {
  const mode = elModeVal();
  const gran = elGranVal();
  const cnt = +($("#elCount") || {}).value || 1;
  const m = elModeMeta(mode);
  $("#elModeNote").innerHTML = (mode === "restage")
    ? `<b>${m.name}</b>：<b>Allowed:</b> new tabletop, florals, props, lighting and framing, plus a recomposed arrangement — <b>but each element's own design is untouched, character for character</b>. Pair with "full set" scope to get a single, freshly shot image of the whole suite.`
    : `<b>${m.name}</b>：Cleanup and faithful restoration only (hands / clutter / watermarks / sharpness / geometry / lighting) — <b>no design information is altered</b>. Pair with "per element" scope to get one clean image per element.`;
  const cap = Math.max(1, 15 - elRefCount());
  $("#elGranNote").innerHTML = (gran === "set")
    ? `All elements composed into one scene image — ideal as the first / last frame of a whole-set, card-by-card reveal video.`
    : `One image per element, up to ${cap} (Ark limit: reference images + generated images ≤ 15). The model decides the actual count from how many elements it identifies.`;
  const ref = $("#elRef").value;
  const refDesc = ref ? `Reference specified: <code>${basename(ref)}</code>` : `Automatically uses step 1 uploads (up to 3)`;
  $("#elRefNote").innerHTML = `${refDesc} · quality <b>${$("#elSize").value}</b> · up to <b>${cnt} images</b>` +
    `(the model decides the count; 4K about ¥0.22 each, less at 2K/3K)`;
}

function renderElGrid() {
  const g = $("#elGrid");
  if (!g) return;
  const els = S.project.elements || [];
  if (!els.length) {
    g.innerHTML = `<div class="empty" style="grid-column:1/-1">No reference assets yet. Choose a mode and scope, then click "Generate Element Assets".</div>`;
    return;
  }
  g.innerHTML = els.map(e => `
    <div class="thumb el">
      <img src="/media/${S.project.id}/elements/${encodeURIComponent(e.file)}" loading="lazy" title="${e.file}">
      <div class="cap">${e.mode === "restage" ? "Restage" : "Fidelity"} · ${e.granularity === "set" ? "full set" : "element"} · ${e.size || ""}</div>
    </div>`).join("");
}

function renderElementStep() {
  // 换项目：重置指令词编辑标记与模式（避免把上个项目的编辑带过来）
  const pid = (S.project || {}).id || null;
  if (S.elPid !== pid) {
    S.elPid = pid;
    S.elEdited = false;
    S.elMode = null;
    _elLastMode = null;
  }
  renderElMode();
  renderElGranOptions();
  renderElRefOptions();
  renderElSizeOptions();
  renderElArkState();
  if (!$("#elPrompt").value.trim()) { S.elEdited = false; syncElPrompt(false); }
  renderElNotes();
  renderElGrid();
  renderProcOptions();
}

if ($("#elPrompt")) $("#elPrompt").addEventListener("input", () => { S.elEdited = true; });
if ($("#elCount")) $("#elCount").addEventListener("change", renderElNotes);
if ($("#btnElReset")) $("#btnElReset").addEventListener("click", () => {
  syncElPrompt(true);
  toast("Restored the official mode prompt (current mode + scope)");
});
if ($("#btnElCopy")) $("#btnElCopy").addEventListener("click", () => copyText($("#elPrompt").value));

$("#btnElGenerate").addEventListener("click", () => {
  if (!S.project) return toast("Select a project first");
  if (!(S.project.uploads || []).length) return toast("Go back to step 1 and upload your product photos first");
  const mode = elModeVal(), gran = elGranVal();
  const ref = $("#elRef").value;
  const prompt = $("#elPrompt").value.trim();
  const extra = [$("#elExtra").value.trim(), $("#elShotList").value.trim()]
    .filter(Boolean).join("\n");
  const box = $("#elJob");
  box.innerHTML = `<div class="job-msg">AI processing job submitted (${elModeMeta(mode).name} · ${gran === "set" ? "full set" : "per element"})…` +
    `Image sets usually take 30–120 s — you can work on another step meanwhile</div>
    <div class="progress"><div style="width:5%"></div></div>`;
  $("#btnElGenerate").disabled = true;
  api("/api/elements/generate", {
    project: S.project.id, mode, granularity: gran,
    images: ref ? [ref] : undefined,
    prompt: (S.elEdited && prompt) ? prompt : undefined,   // 未编辑则由服务端按模式+粒度组装
    extra: extra || undefined,
    size: $("#elSize").value, max_images: +$("#elCount").value,
    output_format: $("#elFmt").value,
  }).then(({ job }) => pollJob(job, box, async (res) => {
    $("#btnElGenerate").disabled = false;
    box.innerHTML = `<div class="job-msg">Generation complete ✓ ${res.count} image(s) · ${res.mode_name} · ` +
      `${res.model} · ${res.size} · ${res.elapsed}s elapsed` +
      (res.note ? `<br><span class="hint">${res.note}</span>` : "") +
      `<br><span class="hint">References: ${(res.used_refs || []).join("、")}</span></div>`;
    const { project } = await api("/api/project?id=" + S.project.id);
    S.project = project;
    renderElementStep();
    toast(`${res.count} reference assets generated — step 3 will use them by default`);
  })).catch(e => {
    $("#btnElGenerate").disabled = false;
    box.innerHTML = `<div class="job-error">✕ ${e.message}</div>`;
  });
});

/* ---- 降级通道：本地算子（零 AI 算力） ---- */
// 全部可用素材：元素素材（AI 产物）优先 → 上传原图 → 本地处理产物
function allSources() {
  const list = [];
  (S.project.elements || []).forEach(e => {
    list.push({ path: e.path, file: e.file,
      url: `/media/${S.project.id}/elements/${encodeURIComponent(e.file)}`,
      verdict: e.granularity === "set" ? "Full-set reference" : "Element reference" });
  });
  (S.project.uploads || []).forEach(e => {
    list.push({ path: e.path, file: e.file,
      url: `/media/${S.project.id}/uploads/${encodeURIComponent(e.file)}`, verdict: "" });
  });
  Object.entries(S.project.processed || {}).forEach(([src, e]) => {
    list.push({ path: e.path, file: e.file,
      url: `/media/${S.project.id}/processed/${encodeURIComponent(e.file)}`, verdict: "Local processing" });
  });
  return list;
}

function renderProcOptions() {
  const sel = $("#procFile");
  if (!sel) return;
  const srcs = allSources();
  const cur = sel.value;
  sel.innerHTML = srcs.map(s =>
    `<option value="${s.path}">${s.file}${s.verdict ? "（" + s.verdict + "）" : ""}</option>`).join("");
  if (cur && Array.from(sel.options).some(o => o.value === cur)) sel.value = cur;
}

$("#btnProcess").addEventListener("click", async () => {
  const path = $("#procFile").value;
  if (!path) return toast("Select an asset");
  const ops = $$('input[type="checkbox"]:checked')
    .filter(c => c.closest("fieldset.ops")).map(c => c.value);
  if (!ops.length) return toast("Select at least one local operator");
  const body = { project: S.project.id, file: path, ops };
  if (ops.includes("crop")) body.crop_percent = [
    +$("#cropX0").value, +$("#cropY0").value, +$("#cropX1").value, +$("#cropY1").value];
  const btn = $("#btnProcess"); btn.disabled = true; btn.textContent = "Processing…";
  $("#procResult").innerHTML = `<div class="job-msg">Processing locally, one moment…</div>`;
  try {
    const { result, entry, project } = await api("/api/process", body);
    S.project = project;
    $("#procResult").innerHTML = `
      <div class="hh-job">
        <div class="hd"><b>Local processing complete</b><span class="badge ok">${result.size[0]}×${result.size[1]}</span></div>
        <div class="hint">Operators: ${result.ops.join(" → ")}</div>
        <div class="before-after">
          <figure><img src="${(allSources().find(s => s.path === path) || {}).url || ""}"><figcaption>Before</figcaption></figure>
          <figure><img src="/media/${S.project.id}/processed/${encodeURIComponent(entry.file)}"><figcaption>After</figcaption></figure>
        </div>
      </div>`;
    renderElementStep();
    toast("Local processing complete — output added to the reference candidates");
  } catch (e) {
    $("#procResult").innerHTML = `<div class="job-error">✕ ${e.message}</div>`;
  }
  btn.disabled = false; btn.textContent = "Run local processing";
});
function basename(p) { return p.replace(/\\/g, "/").split("/").pop(); }

/* ================= 步骤3 视频生成（基础运镜 + AI 手持视频，同屏） ================= */
function renderVideoStep() {
  renderCmSources();     // ① 基础运镜：源素材
  renderCmSeqFixed();    // ① 固定五镜序列展示
  renderCmMoveInfo();    // ① 时长合计
  renderShowcase();      // ② 分段手持展示（识别 → 排序 → 逐段生成 → Hero）
  renderFirstFrame();    // ②b 旧版 AI 手持：首帧来源
  renderLastFrame();     // ②b 旧版 AI 手持：尾帧来源（可选，首尾帧模式）
  renderHhResOptions();
  renderHhArkState();
  renderHandheldLibrary();
  renderStoryboards();   // ③ 可选：Seedream 分镜图
}

/* ---- ② 分段手持展示（一镜一卡 · 2026-09-24 新口径） ---- */
function scSetCandidates() {
  const els = (S.project.elements || []).filter(e => e.granularity === "set");
  const ups = S.project.uploads || [];
  return { els, ups };
}
function renderShowcase() {
  const sel = $("#scSource");
  if (!sel) return;
  const { els, ups } = scSetCandidates();
  const cur = sel.value;
  let html = "";
  if (els.length) html += `<optgroup label="Full-set element assets (step 2 output · recommended)">` +
    els.map(e => `<option value="${e.path}">${e.file}</option>`).join("") + `</optgroup>`;
  if (ups.length) html += `<optgroup label="Step 1 uploads — full set">` +
    ups.map(e => `<option value="${e.path}">${e.file}</option>`).join("") + `</optgroup>`;
  sel.innerHTML = html || `<option value="">(No assets yet — upload your product photos first)</option>`;
  const valid = Array.from(sel.options).some(o => o.value === cur && cur);
  sel.value = valid ? cur : (els.length ? els[els.length - 1].path :
    (sel.options[0] || {}).value || "");
  renderScPlan();
  renderScSegs();
}
function scPlan() { return (S.project || {}).showcase || null; }

function runScPlan() {
  const src = $("#scSource").value;
  if (!src) return toast("Select the set source image");
  const depth = (document.querySelector('input[name="scdepth"]:checked') || {}).value || "fast";
  const box = $("#scPlanJob");
  box.innerHTML = `<div class="job-msg">${depth === "deep" ? "Deep identification (~100 s)…" : "Identifying components (~15 s)…"}</div>
    <div class="progress"><div style="width:8%"></div></div>`;
  api("/api/showcase/plan", { project: S.project.id, src, depth })
    .then(({ job }) => pollJob(job, box, (res) => {
      box.innerHTML = `<div class="job-msg">Identification complete ✓ ${res.summary || ""}</div>`;
      S.project.showcase = res.plan;
      renderScPlan();
    })).catch(e => box.innerHTML = `<div class="job-error">✕ ${e.message}</div>`);
}

function renderScPlan() {
  const plan = scPlan();
  const q = $("#scQuality"), list = $("#scPlanList"), genRow = $("#scGenRow"), heroRow = $("#scHeroRow");
  if (!list) return;
  if (!plan) {
    q.innerHTML = ""; list.innerHTML = ""; genRow.hidden = true; heroRow.hidden = true;
    list.innerHTML = `<div class="empty">Not identified yet. Choose the set image, then click "1 · Identify components & order".</div>`;
    return;
  }
  const qi = { good: ["ok", "Good — usable directly as the first frame for each clip"], fair: ["warn", "Fair — some components are occluded or overlapping"],
    poor: ["bad", "Poor — generate a storyboard before generating video"] }[plan.image_quality] || ["warn", "Unknown"];
  q.innerHTML = `<div class="banner ${qi[0]}">Quality check: ${qi[1]}${plan.scene ? ` · scene: ${plan.scene}` : ""}` +
    (plan.issues && plan.issues.length ? `<br>Notes: ${plan.issues.join("；")}` : "") + `</div>`;
  list.innerHTML = `<div class="sc-plan">
    ${plan.components.map((c, i) => `
    <div class="seg sc-row" data-id="${c.id}">
      <label class="seg-pick"><input type="checkbox" ${c.include !== false ? "checked" : ""}
        onchange="scToggle('${c.id}', this.checked)"></label>
      <span class="idx">${i + 1}</span>
      <span class="name"><b>${c.label || c.label_zh}</b> <span class="hint">${c.pos_en || c.pos}${c.note ? " · " + c.note : ""}</span></span>
      <select class="sc-dur" onchange="scDur('${c.id}', this.value)">
        ${[1.5, 2, 2.5, 3, 3.5, 4].map(v => `<option value="${v}" ${+c.dur === v ? "selected" : ""}>${v}s</option>`).join("")}
      </select>
      <button class="btn btn-mini" onclick="scMove('${c.id}', -1)" title="Move up">↑</button>
      <button class="btn btn-mini" onclick="scMove('${c.id}', 1)" title="Move down">↓</button>
      <button class="btn btn-mini" onclick="scDel('${c.id}')" title="Delete this clip">✕</button>
    </div>`).join("")}
  </div>`;
  genRow.hidden = false;
  heroRow.hidden = !(S.project.handheld_videos || []).some(v => v.kind === "showcase_seg");
}
function scRow(id) { return scPlan().components.find(c => c.id === id); }
function scToggle(id, on) { scRow(id).include = on; }
function scDur(id, v) { scRow(id).dur = +v; }
function scDel(id) {
  const p = scPlan();
  p.components = p.components.filter(c => c.id !== id);
  p.components.forEach((c, i) => c.order = i + 1);
  renderScPlan();
}
function scMove(id, dir) {
  const p = scPlan();
  const i = p.components.findIndex(c => c.id === id);
  const j = i + dir;
  if (i < 0 || j < 0 || j >= p.components.length) return;
  [p.components[i], p.components[j]] = [p.components[j], p.components[i]];
  p.components.forEach((c, k) => c.order = k + 1);
  renderScPlan();
}
async function saveScPlan(silent) {
  const p = scPlan();
  if (!p) return;
  const { project } = await api("/api/showcase/update",
    { project: S.project.id, components: p.components });
  S.project = project;
  if (!silent) toast("Plan saved");
}
$("#btnScPlan").addEventListener("click", runScPlan);
if ($("#btnScSavePlan")) $("#btnScSavePlan").addEventListener("click", () => saveScPlan().catch(e => toast(e.message)));

$("#btnScGenerate").addEventListener("click", async () => {
  const plan = scPlan();
  if (!plan) return toast("Identify the components first");
  const ids = plan.components.filter(c => c.include !== false).map(c => c.id);
  if (!ids.length) return toast("Keep at least one component clip");
  const res = $("#scResolution").value;
  const secs = plan.components.filter(c => ids.includes(c.id)).reduce((s, c) => s + (+c.dur || 2.5), 0);
  const box = $("#scGenJob");
  box.innerHTML = `<div class="job-msg">Clip generation submitted: ${ids.length} clip(s) · 4s generated then trimmed · about ${secs.toFixed(1)}s total · estimated ${Math.max(2, ids.length * 2)}–${ids.length * 4} min — you can work on another step meanwhile…</div>
    <div class="progress"><div style="width:3%"></div></div>`;
  try { await saveScPlan(true); } catch (e) {}
  api("/api/showcase/generate", { project: S.project.id, ids, resolution: res })
    .then(({ job }) => pollJob(job, box, async (r) => {
      const bad = (r.check || []).filter(x => !x.ok || x.warn);
      box.innerHTML = `<div class="job-msg">Clip generation complete ✓ ${r.summary || ""}</div>` +
        (bad.length ? `<div class="job-error">Consistency warning: ${bad.map(x => x.file + "（" + (x.warn || x.reason) + "）").join("；")}</div>`
          : `<div class="hint">Consistency check passed: ${(r.check || []).length} clip(s) share the same resolution and frame rate</div>`);
      const { project } = await api("/api/project?id=" + S.project.id);
      S.project = project;
      renderScSegs(); renderScPlan();
    })).catch(e => box.innerHTML = `<div class="job-error">✕ ${e.message}</div>`);
});

$("#btnScHero").addEventListener("click", () => {
  const move = $("#scHeroMove").value, secs = $("#scHeroSecs").value;
  const box = $("#scHeroJob");
  box.innerHTML = `<div class="job-msg">Rendering the Hero ending (geometric camera move, ~1–2 min)…</div>
    <div class="progress"><div style="width:5%"></div></div>`;
  api("/api/showcase/hero", { project: S.project.id, move, secs: +secs })
    .then(({ job }) => pollJob(job, box, async (r) => {
      box.innerHTML = `<div class="job-msg">Hero complete ✓ ${r.summary || ""} · ${r.elapsed}s elapsed</div>`;
      const { project } = await api("/api/project?id=" + S.project.id);
      S.project = project;
      renderScSegs();
    })).catch(e => box.innerHTML = `<div class="job-error">✕ ${e.message}</div>`);
});

function renderScSegs() {
  const box = $("#scSegs");
  if (!box) return;
  const segs = (S.project.handheld_videos || []).filter(v => v.kind === "showcase_seg" || v.kind === "showcase_hero");
  heroRowShow(segs.length > 0);
  box.innerHTML = segs.length ? `<h3>Clip outputs (stitched in order)</h3>` + segs
    .slice().sort((a, b) => (a.order || 0) - (b.order || 0) || String(a.created).localeCompare(String(b.created)))
    .map(v => `
    <div class="seg">
      <span class="idx">${v.kind === "showcase_hero" ? "★" : (v.order || "?")}</span>
      <span class="name">${v.name || v.file} <span class="tag">${v.kind === "showcase_hero" ? "Hero" : "Segment"}</span></span>
      <video class="result" controls style="max-height:220px"
        src="/media/${S.project.id}/handheld/${encodeURIComponent(basename(v.file))}"></video>
    </div>`).join("") : "";
}
function heroRowShow(on) { const r = $("#scHeroRow"); if (r) r.hidden = !on; }


/* ---- ① 基础运镜 ---- */
function renderCmSources() {
  const sel = $("#cmSource");
  const cur = sel.value;
  const els = S.project.elements || [];
  const ups = S.project.uploads || [];
  const prc = Object.values(S.project.processed || {});
  let html = "";
  if (els.length) html += `<optgroup label="Element assets (step 2 output · default source)">` +
    els.map(e => `<option value="${e.path}">${e.file}（${e.granularity === "set" ? "full set" : "element"}）</option>`).join("") +
    `</optgroup>`;
  if (ups.length) html += `<optgroup label="Step 1 uploads (fallback)">` +
    ups.map(e => `<option value="${e.path}">${e.file}</option>`).join("") + `</optgroup>`;
  if (prc.length) html += `<optgroup label="Local processing output">` +
    prc.map(e => `<option value="${e.path}">${e.file}</option>`).join("") + `</optgroup>`;
  sel.innerHTML = html || `<option value="">(No assets yet)</option>`;
  const valid = Array.from(sel.options).some(o => o.value === cur && cur);
  // 默认来源 = 最新一张元素素材；无元素素材时回落到第一项（上传原图）
  sel.value = valid ? cur
    : (els.length ? els[els.length - 1].path : (sel.options[0] || {}).value || "");
  // 已渲染列表
  const cms = S.project.cameramove || [];
  const seg = $("#cmRenderOut");
  if (cms.length) {
    seg.innerHTML = `<h3>Rendered output</h3>` + cms.map(v => `
      <video class="result" controls src="/media/${S.project.id}/cameramove/${encodeURIComponent(basename(v.file))}"></video>
      <div class="file-path">${v.file}</div>`).join("");
  } else {
    seg.innerHTML = "";
  }
  renderCmMulti();   // 同步多图拼接素材池（默认单图模式时仅更新可见性）
}
// 锚点输入（3 个，百分比）
(function initAnchors() {
  const names = ["Primary anchor (push in / close-up)", "Secondary anchor (pan)", "Tertiary anchor (detail)"];
  $("#anchorInputs").innerHTML = names.map((n, i) => `
    <div class="anch"><b>${n}</b><br>
      X <input type="number" id="an${i}x" placeholder="auto" min="0" max="100">
      Y <input type="number" id="an${i}y" placeholder="auto" min="0" max="100"> %
    </div>`).join("");
})();
function getAnchorsPct() {
  const pts = [];
  for (let i = 0; i < 3; i++) {
    const x = $("#an" + i + "x").value, y = $("#an" + i + "y").value;
    if (x !== "" && y !== "") pts.push([+x, +y]);
  }
  return pts.length ? pts : null;
}
function cmMoves() { return (S.config || {}).camera_moves || []; }
function cmMoveById(id) { return cmMoves().find(x => x.id === id) || null; }

/* ---- ① 固定五镜序列（用户口径，不再勾选）：CM-01→CM-04→CM-02→CM-07→CM-05 ---- */
const CM_STANDARD = ["cm01", "cm04", "cm02", "cm07", "cm05"];

function renderCmSeqFixed() {
  const box = $("#cmSeqFixed");
  if (!box) return;
  const moves = cmMoves();
  if (!moves.length) return;                      // 镜头库未就绪，等 /api/config 回来再填
  if (box.dataset.ready) return;
  box.innerHTML = CM_STANDARD.map((id, i) => {
    const m = cmMoveById(id);
    const nm = m ? m.name.split(" ") : [id.toUpperCase()];
    return `<div class="cm-chip on" title="${m ? "Suits: " + m.suits + " · Stability: " + m.stability : ""}">
      <span class="ord">${i + 1}</span><b>${nm[0]}</b><em>${nm.slice(1).join(" ")}</em><i>${m ? m.secs : 2.5}s</i></div>`;
  }).join("");
  box.dataset.ready = "1";
}

function cmSecsVal() {
  const v = $("#cmSecs") ? $("#cmSecs").value : "";
  return v ? +v : 2.5;
}

function renderCmMoveInfo() {
  const el = $("#cmMoveInfo");
  if (!el) return;
  const per = cmSecsVal();
  const multi = $("#cmMulti") && $("#cmMulti").checked;
  const rows = multi ? collectCmMulti() : [];
  if (multi) {
    const n = rows.length;
    const total = n * per + 0.6;
    el.innerHTML = `Multi-image sequence — <b>${n} images</b> · ${per}s per shot · <b>about ${total.toFixed(1)}s total</b> (includes a 0.6s freeze at the end)` +
      ` · about ${Math.max(2, Math.round(total * 15 / 60))}–${Math.max(3, Math.round(total * 25 / 60) + 1)} min to render`;
    return;
  }
  const total = CM_STANDARD.length * per;
  const order = CM_STANDARD.map((id, i) => {
    const m = cmMoveById(id);
    return `${i + 1}.${m ? m.name.split(" ")[0] : id}`;
  }).join(" ");
  el.innerHTML = `Fixed sequence <b>${order}</b> · ${per}s per shot · <b>about ${total.toFixed(1)}s total</b> (hard cuts between shots)` +
    ` · about ${Math.max(2, Math.round(total * 15 / 60))}–${Math.max(3, Math.round(total * 25 / 60) + 1)} min to render`;
}

/* ---- 多图拼接：池 = 上传原图（优先）+ 元素素材，逐张分配镜头 ---- */
function cmMultiPool() {
  const ups = (S.project.uploads || []).map(e => ({ ...e, grp: "Uploads" }));
  const els = (S.project.elements || []).map(e => ({ ...e, grp: "Element assets" }));
  return ups.concat(els);
}
function renderCmMulti() {
  const box = $("#cmMultiList");
  const on = $("#cmMulti") && $("#cmMulti").checked;
  if (!box) return;
  box.hidden = !on;
  $("#cmSource").closest(".form-row").hidden = on;
  if (!on) { renderCmMoveInfo(); return; }
  const pool = cmMultiPool();
  // 池未变则保留用户已分配的镜头（避免切步骤回来被重置）
  const sig = (S.project.id || "") + "::" + pool.map(e => e.path).join("|");
  if (box.dataset.sig === sig) { renderCmMoveInfo(); return; }
  box.dataset.sig = sig;
  if (pool.length < 2) {
    box.innerHTML = `<p class="hint">Only ${pool.length} asset(s) available — a multi-image sequence needs at least 2. <b>Upload more photos in step 1 first.</b></p>`;
    renderCmMoveInfo();
    return;
  }
  const moves = cmMoves();
  const opts = moves.map(m => `<option value="${m.id}">${m.name}（${m.secs}s）</option>`).join("");
  box.innerHTML = `<div class="mrows">` + pool.map((e, i) => {
    const def = CM_STANDARD[i % CM_STANDARD.length];
    const url = `/media/${S.project.id}/${e.grp === "Element assets" ? "elements" : "uploads"}/${encodeURIComponent(e.file)}`;
    return `<div class="mrow" data-path="${e.path}">
      <input type="checkbox" class="mOn" checked title="Include in the sequence">
      <span class="mno">${i + 1}</span>
      <img src="${url}" loading="lazy" title="${e.file}">
      <span class="mf">${e.file} <span class="tag">${e.grp}</span></span>
      <select class="mMove">${opts}</select>
      <span class="mdur"></span>
    </div>`;
  }).join("") + `</div>`;
  // 默认镜头 = 固定五镜轮转
  box.querySelectorAll(".mrow").forEach((row, i) => {
    const sel = row.querySelector(".mMove");
    sel.value = CM_STANDARD[i % CM_STANDARD.length];
    const sync = () => {
      const m = cmMoveById(sel.value);
      row.querySelector(".mdur").textContent = (m ? m.secs : 2.5) + "s";
      row.classList.toggle("off", !row.querySelector(".mOn").checked);
      renderCmMoveInfo();
    };
    sel.addEventListener("change", sync);
    row.querySelector(".mOn").addEventListener("change", sync);
    sync();
  });
}
function collectCmMulti() {
  const out = [];
  document.querySelectorAll("#cmMultiList .mrow").forEach(row => {
    if (!row.querySelector(".mOn").checked) return;
    out.push({ path: row.dataset.path, move: row.querySelector(".mMove").value });
  });
  return out;
}
if ($("#cmMulti")) $("#cmMulti").addEventListener("change", renderCmMulti);

if ($("#cmSecs")) $("#cmSecs").addEventListener("change", renderCmMoveInfo);

$("#btnCmRender").addEventListener("click", () => {
  const multi = $("#cmMulti") && $("#cmMulti").checked;
  const box = $("#cmJob");
  if (multi) {
    const rows = collectCmMulti();
    if (rows.length < 2) return toast("A multi-image sequence needs at least 2 selected photos");
    const secs = cmSecsVal();
    const eta = rows.length * secs + 0.6;
    const lo = Math.max(1, Math.round(eta * 14 / 60));
    const hi = Math.max(lo + 1, Math.round(eta * 25 / 60) + 1);
    box.innerHTML = `<div class="job-msg">Multi-image sequence job submitted (${rows.length} images · one camera move each · about ${eta.toFixed(1)}s)，` +
      `Estimated ${lo}–${hi} min — you can work on another step meanwhile…</div>
      <div class="progress"><div style="width:3%"></div></div>`;
    api("/api/cameramove/render", {
      project: S.project.id, sources: rows.map(r => r.path), moves: rows.map(r => r.move), secs,
    }).then(({ job }) => pollJob(job, box, async (res) => {
      box.innerHTML = `<div class="job-msg">Render complete ✓ ${res.label || ""} · ${res.elapsed}s elapsed · ${res.secs}s final cut` +
        ((res.shots || []).length > 1 ? `<br><span class="hint">Shot sequence: ${res.shots.join(" → ")} (hard cuts)</span>` : "") +
        `</div>`;
      const { project } = await api("/api/project?id=" + S.project.id);
      S.project = project;
      renderCmSources();
    })).catch(e => box.innerHTML = `<div class="job-error">✕ ${e.message}</div>`);
    return;
  }
  const src = $("#cmSource").value;
  if (!src) return toast("Select a source image");
  const secs = cmSecsVal();
  const eta = CM_STANDARD.length * secs;
  const lo = Math.max(1, Math.round(eta * 14 / 60));
  const hi = Math.max(lo + 1, Math.round(eta * 25 / 60) + 1);
  box.innerHTML = `<div class="job-msg">Render job submitted (standard five-shot sequence · ${CM_STANDARD.join("→")} · about ${eta.toFixed(1)}s)，` +
    `Estimated ${lo}–${hi} min — you can work on another step meanwhile…</div>
    <div class="progress"><div style="width:3%"></div></div>`;
  api("/api/cameramove/render", {
    project: S.project.id, src, moves: CM_STANDARD.slice(), secs,
    anchors_pct: getAnchorsPct(),
  }).then(({ job }) => pollJob(job, box, async (res) => {
    box.innerHTML = `<div class="job-msg">Render complete ✓ ${res.label || ""} · ${res.elapsed}s elapsed · ${res.secs}s final cut` +
      ((res.shots || []).length > 1 ? `<br><span class="hint">Shot sequence: ${res.shots.join(" → ")} (hard cuts)</span>` : "") +
      `</div>`;
    const { project } = await api("/api/project?id=" + S.project.id);
    S.project = project;
    renderCmSources();
  })).catch(e => box.innerHTML = `<div class="job-error">✕ ${e.message}</div>`);
});

/* ---- ② AI 手持展示视频：首帧来源 ---- */
function firstFrameOptions() {
  const els = (S.project.elements || []).map(e => ({ ...e, group: "Element assets" }));
  // 整套参考图排在前面（贴合「整套逐张展示」口径），其余元素素材按生成先后
  const setEls = els.filter(e => e.granularity === "set").reverse();
  const itemEls = els.filter(e => e.granularity !== "set").reverse();
  const ups = (S.project.uploads || []).map(e => ({ ...e, group: "Step 1 uploads" }));
  const prc = Object.values(S.project.processed || {}).map(e => ({ ...e, group: "Local processing output" }));
  const sbs = (S.project.storyboards || []).map(e => ({ ...e, group: "Existing storyboards (Seedream / uploaded)" }));
  return { setEls, itemEls, ups, prc, sbs };
}
function ffOptgroups(o) {
  let html = "";
  if (o.setEls.length) html += `<optgroup label="Element assets · full set (step 2 output · preferred)">` +
    o.setEls.map(e => `<option value="${e.path}">${e.file} · full set</option>`).join("") + `</optgroup>`;
  if (o.itemEls.length) html += `<optgroup label="Element assets · per element (step 2 output)">` +
    o.itemEls.map(e => `<option value="${e.path}">${e.file} · element</option>`).join("") + `</optgroup>`;
  if (o.ups.length) html += `<optgroup label="Step 1 uploads (fallback)">` +
    o.ups.map(e => `<option value="${e.path}">${e.file}</option>`).join("") + `</optgroup>`;
  if (o.prc.length) html += `<optgroup label="Local processing output">` +
    o.prc.map(e => `<option value="${e.path}">${e.file}</option>`).join("") + `</optgroup>`;
  if (o.sbs.length) html += `<optgroup label="Existing storyboards (Seedream / uploaded)">` +
    o.sbs.map(e => `<option value="${e.path}">${e.file}</option>`).join("") + `</optgroup>`;
  return html;
}
function renderFirstFrame() {
  const sel = $("#hhFirstFrame");
  const cur = sel.value;
  const o = firstFrameOptions();
  let html = `<option value="">(Auto: prefers the highest-resolution step 2 element asset, falling back to the original upload)</option>` +
    ffOptgroups(o);
  sel.innerHTML = html;
  if (cur && Array.from(sel.options).some(v => v.value === cur)) sel.value = cur;
  renderFirstFrameNote();
}
function renderFirstFrameNote() {
  const ups = (S.project.uploads || []).length;
  const els = (S.project.elements || []).length;
  const sbs = (S.project.storyboards || []).length;
  const v = $("#hhFirstFrame").value;
  const note = $("#hhFirstFrameNote");
  if (v) {
    const isEl = (S.project.elements || []).some(e => e.path === v);
    note.innerHTML = `First frame selected: <code>${basename(v)}</code>${isEl ? " (step 2 element assets)" : ""} · this image becomes the video's first frame and the product's sole visual reference`;
  } else if (els || ups || sbs) {
    note.innerHTML = `Auto: prefers the <b>${els} element asset(s)</b>, favouring full-set references and the highest resolution` +
      `${els ? "" : `, falling back to ${ups} upload(s)${sbs ? ` + ${sbs} storyboard(s)` : ""}`} — the result will report which file was actually used.`;
  } else {
    note.innerHTML = `<span class="red">No usable assets in this project — go back to step 1 and upload product photos.</span>`;
  }
}
$("#hhFirstFrame").addEventListener("change", () => { renderFirstFrameNote(); renderLastFrame(); });

/* ---- ② AI 手持展示视频：尾帧来源（可选，首尾帧模式） ---- */
function renderLastFrame() {
  const sel = $("#hhLastFrame");
  if (!sel) return;
  const cur = sel.value;
  const o = firstFrameOptions();
  let html = `<option value="">(None: first frame only — the model chooses the camera move)</option>` +
    `<option value="auto">(Auto: prefers element assets — the highest-resolution one that differs from the first frame)</option>` +
    ffOptgroups(o);
  sel.innerHTML = html;
  if (cur && Array.from(sel.options).some(v => v.value === cur)) sel.value = cur;
  renderLastFrameNote();
}
function renderLastFrameNote() {
  const note = $("#hhLastFrameNote");
  if (!note) return;
  const v = $("#hhLastFrame").value;
  const ff = $("#hhFirstFrame").value;
  if (v && v === ff) {
    note.innerHTML = `<span class="red">Last frame matches the first: only the first frame will be sent (Ark allows identical frames, but there is nothing to transition to). Choose another image — e.g. the shell as first frame and the insert as last frame.</span>`;
  } else if (v === "auto") {
    note.innerHTML = `<b>First/last-frame mode</b>: prefers an <b>element asset</b> that differs from the first frame as the last frame (falls back to an original upload if there are no element assets), so the video transitions naturally from the first to the last frame — e.g. closing the shell → opening the insert.`;
  } else if (v) {
    note.innerHTML = `<b>First/last-frame mode</b>: first frame <code>${basename(ff || "auto-detected")}</code> → last frame <code>${basename(v)}</code> — the model generates a natural transition between the two.`;
  } else {
    note.innerHTML = `No last frame means ordinary image-to-video (anchored on the first frame only, with the model choosing the camera move). Pick one when you want an open/close action such as &ldquo;close → open&rdquo;.`;
  }
}
$("#hhLastFrame").addEventListener("change", renderLastFrameNote);

/* ---- ② AI 手持展示视频：参数与生成 ---- */
$("#hhPrompt").value = P.video_handheld;
$("#btnHhReset").addEventListener("click", () => {
  $("#hhPrompt").value = P.video_handheld;
  toast("Restored the official video prompt (Seedance 2.0 mini general template)");
});
$("#hhDuration").addEventListener("change", () => {
  $("#hhDurationCustom").style.display =
    $("#hhDuration").value === "custom" ? "inline-block" : "none";
});
function hhModeVal() { return document.querySelector('input[name="hhmode"]:checked').value; }
$$('input[name="hhmode"]').forEach(r => r.addEventListener("change", () => {
  $("#hhArkBox").style.display = hhModeVal() === "ark" ? "" : "none";
  $("#btnHhGenerate").style.display = hhModeVal() === "ark" ? "" : "none";
}));
function hhMeta() {
  const v = (S.config || {}).video_api || {};
  return v.meta || { dur: [4, 15], res: ["480p", "720p"], id: v.effective_model || "" };
}
function renderHhResOptions() {
  const m = hhMeta();
  const sel = $("#hhResolution"), cur = sel.value;
  sel.innerHTML = (m.res || ["720p"]).map(r =>
    `<option value="${r}">${r}${r === "720p" ? " (project default)" : ""}${r === "480p" ? " (draft check, cheapest)" : ""}</option>`).join("");
  sel.value = (m.res || []).includes(cur) ? cur : ((m.res || []).includes("720p") ? "720p" : m.res[0]);
  const lo = (m.dur || [4, 15])[0], hi = (m.dur || [4, 15])[1];
  $("#hhDurationCustom").min = lo; $("#hhDurationCustom").max = hi;
}
function renderHhArkState() {
  const v = (S.config || {}).video_api || {};
  const box = $("#hhArkState");
  if (v.api_key_set) {
    const m = hhMeta();
    box.className = "banner ok";
    box.innerHTML = `Volcengine Ark ready · base <code>${v.effective_endpoint || "default"}</code> · model <code>${v.effective_model || ""}</code>
      · supports ${(m.res || []).join("/")} · ${(m.dur || [])[0]}–${(m.dur || [])[1]}s<br>
      Runs as an async job after submission, usually 1–4 min to finish. Out-of-range resolution or duration is clamped to the model's capability range (noted in the result).`;
  } else {
    box.className = "banner warn";
    box.innerHTML = `ARK_API_KEY is not configured, so direct AI generation is unavailable. Go to <b>Settings → Video generation API</b> and enter your Volcengine Ark long-lived API key,
      or switch to "Brief + upload finished clip" mode.`;
  }
}
function renderHandheldLibrary() {
  // 任务单列表
  const jobs = S.project.handheld_jobs || [];
  let html = jobs.length ? jobs.map(j => `
    <div class="hh-job">
      <div class="hd"><b>Brief ${j.id}</b>
        <span class="badge ${j.status === "done" ? "ok" : "pending"}">${j.status === "done" ? "Done" : "Pending"}</span>
        <span class="hint">Duration ${j.duration}s</span></div>
      <pre>${j.prompt}</pre>
      <div class="btn-row">
        <button class="btn" onclick="copyText(${JSON.stringify(j.prompt).replace(/"/g, "&quot;")})">Copy prompt</button>
        <a class="btn" href="${assetUrl(j.image)}" target="_blank">Open first frame</a>
        ${j.video ? `<video class="result" controls src="/media/${S.project.id}/handheld/${encodeURIComponent(j.video.file)}"></video>` : ""}
      </div>
    </div>`).join("") : "";
  // 已生成/上传视频列表
  const vids = S.project.handheld_videos || [];
  if (vids.length) {
    html += `<h3>Handheld video library</h3>` + vids.map(v => `
      <video class="result" controls src="/media/${S.project.id}/handheld/${encodeURIComponent(v.file)}"></video>
      <div class="file-path">${v.file}${v.kind === "ai_ark" ? ` · Volcengine Ark ${v.model || ""} · ${v.duration || "?"}s · ${v.resolution || ""}` : ""}</div>`).join("");
  }
  $("#hhJobs").innerHTML = html;
}
function copyText(t) { navigator.clipboard.writeText(t).then(() => toast("Prompt copied")); }
// 按素材所在子目录拼出可访问 URL（素材可能来自 elements / uploads / processed / storyboards）
function assetUrl(p) {
  const name = basename(p || "");
  if (!name) return "";
  const pid = S.project.id;
  const hit = (S.project.elements || []).find(e => e.file === name);
  if (hit) return `/media/${pid}/elements/${encodeURIComponent(name)}`;
  if ((S.project.uploads || []).some(e => e.file === name)) return `/media/${pid}/uploads/${encodeURIComponent(name)}`;
  if (Object.values(S.project.processed || {}).some(e => e.file === name)) return `/media/${pid}/processed/${encodeURIComponent(name)}`;
  return `/media/${pid}/storyboards/${encodeURIComponent(name)}`;
}
$("#btnHhCopy").addEventListener("click", () => copyText($("#hhPrompt").value));

function hhDurationVal() {
  let dur = $("#hhDuration").value;
  if (dur === "custom") dur = +$("#hhDurationCustom").value || 5;
  return +dur;
}

$("#btnHhGenerate").addEventListener("click", () => {
  const image = $("#hhFirstFrame").value;   // 允许为空 = 服务端自动识别选图
  const lastImage = ($("#hhLastFrame") || {}).value || "";
  const prompt = $("#hhPrompt").value.trim();
  if (!prompt) return toast("The prompt cannot be empty");
  if (lastImage && lastImage === image && lastImage !== "auto")
    return toast("Last frame matches the first — choose a different image, or clear the last frame");
  if (!(S.project.uploads || []).length && !(S.project.elements || []).length
      && !(S.project.storyboards || []).length)
    return toast("Go back to step 1 and upload product photos, or generate element assets in step 2 to use as the video's first frame");
  const box = $("#hhGenJob");
  box.innerHTML = `<div class="job-msg">Volcengine Ark generation job submitted… (about 1–4 min — please keep this page open)</div>
    <div class="progress"><div style="width:5%"></div></div>`;
  $("#btnHhGenerate").disabled = true;
  api("/api/handheld/generate", {
    project: S.project.id, image, last_image: lastImage, prompt, duration: hhDurationVal(),
    resolution: $("#hhResolution").value, ratio: $("#hhRatio").value,
    generate_audio: $("#hhAudio").checked, watermark: $("#hhWatermark").checked,
  }).then(({ job }) => pollJob(job, box, async (res) => {
    $("#btnHhGenerate").disabled = false;
    const notes = (res.param_notes || []).length
      ? `<div class="job-msg" style="color:#b7791f">Parameters clamped: ${res.param_notes.join("；")}</div>` : "";
    const pick = res.picked_file ? `<div class="job-msg">First frame: ${res.picked_file}${res.auto_picked ? " (auto-detected)" : ""}` +
      (res.last_frame ? ` → last frame: ${res.last_frame}${res.last_auto ? " (auto-detected)" : ""} · first/last-frame mode` : " · first frame only") + `</div>` : "";
    box.innerHTML = `<div class="job-msg">Generation complete ✓ ${res.elapsed}s elapsed · task ${res.task_id} · ${res.model} · ${res.duration}s / ${res.resolution}</div>` + pick + notes;
    const { project } = await api("/api/project?id=" + S.project.id);
    S.project = project;
    renderVideoStep(); renderCompose();
    toast("AI handheld video generated and saved to the library");
  })).catch(e => {
    $("#btnHhGenerate").disabled = false;
    box.innerHTML = `<div class="job-error">✕ ${e.message}</div>`;
  });
});

$("#btnHhCreate").addEventListener("click", async () => {
  try {
    const { job, project } = await api("/api/handheld/job", {
      project: S.project.id, image: $("#hhFirstFrame").value,
      last_image: ($("#hhLastFrame") || {}).value || "",
      prompt: $("#hhPrompt").value, duration: hhDurationVal()
    });
    S.project = project;
    renderHandheldLibrary();
    toast(`Brief created: ${job.id}. Once the video has been generated externally, click "Upload finished video".`);
  } catch (e) { toast("Failed to create the brief: " + e.message); }
});

$("#btnHhUpload").addEventListener("click", () => $("#fileHandheld").click());
$("#fileHandheld").addEventListener("change", async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try {
    const url = `/api/handheld/video?project=${S.project.id}&name=${encodeURIComponent(f.name)}`;
    const r = await fetch(url, { method: "POST", body: f });
    const j = await r.json();
    if (j.error) throw new Error(j.error);
    S.project = j.project;
    renderHandheldLibrary(); renderCompose();
    toast("Handheld video saved to the library");
  } catch (err) { toast("Upload failed: " + err.message); }
  e.target.value = "";
});

/* ---- ③ 可选：Seedream 手持分镜图 ---- */
function sbMeta() {
  const sb = (S.config || {}).storyboard_api || {};
  return sb.meta || { sizes: ["2K", "3K", "4K"], sequential: true, id: sb.effective_model || "" };
}
function renderSbState() {
  const sb = (S.config || {}).storyboard_api || {};
  const box = $("#sbArkState");
  if (sb.api_key_set) {
    box.className = "banner ok";
    const m = sbMeta();
    box.innerHTML = `Seedream ready · model <code>${sb.effective_model || ""}</code> · available quality ${(m.sizes || []).join("/")}
      · Set limit: reference images + generated images ≤ 15<br>
      Output is saved to the library automatically and appears in the "First frame source" list above.`;
  } else {
    box.className = "banner warn";
    box.innerHTML = `ARK_API_KEY is not configured, so AI generation is unavailable. Go to <b>Settings → Storyboard API</b> and enter it (it can be shared with video generation), or upload storyboards you already have.`;
  }
}
let _sbPidLoaded = null, _shotTimer = null;
function renderStoryboards() {
  // 拍摄清单按项目从服务端读取（只在切换项目时载入，避免覆盖正在输入的内容）
  const pid = S.project ? S.project.id : null;
  if (pid !== _sbPidLoaded) {
    _sbPidLoaded = pid;
    $("#sbShotList").value = (S.project && S.project.shot_list) || "";
  }
  // 参考产品图下拉
  const refSel = $("#sbRef");
  const srcs = allSources();
  const cur = refSel.value;
  refSel.innerHTML = `<option value="">(Auto: uses all step 1 uploads, up to 3)</option>` +
    srcs.map(s => `<option value="${s.path}">${s.file}${s.verdict && s.verdict !== "A" ? "（" + s.verdict + "）" : ""}</option>`).join("");
  if (cur) refSel.value = cur;
  // 画质档位随模型收敛
  const sizes = sbMeta().sizes || ["2K", "3K", "4K"];
  const szSel = $("#sbSize"), curSz = szSel.value;
  szSel.innerHTML = sizes.map(s => `<option value="${s}">${s}</option>`).join("");
  szSel.value = sizes.includes(curSz) ? curSz : (sizes[sizes.length - 1] || "4K");
  if (!$("#sbPrompt").value.trim()) $("#sbPrompt").value = P.storyboard_handheld;
  renderSbState();
  // 结果网格
  const g = $("#sbGrid");
  const sbs = S.project.storyboards || [];
  if (!sbs.length) {
    g.innerHTML = `<div class="empty" style="grid-column:1/-1">No storyboards yet. Generate them if needed, or upload storyboards you already have.</div>`;
    return;
  }
  g.innerHTML = sbs.map(e => `
    <div class="thumb">
      <img src="/media/${S.project.id}/storyboards/${encodeURIComponent(e.file)}" loading="lazy">
      <div class="cap">${e.file}${e.kind === "ai_seedream" ? ` · AI ${e.size || ""}` : ""}</div>
    </div>`).join("");
}
$("#btnSbReset").addEventListener("click", () => {
  $("#sbPrompt").value = P.storyboard_handheld;
  toast("Restored the official storyboard prompt");
});
$("#btnSbCopy").addEventListener("click", () => copyText($("#sbPrompt").value));
$("#sbShotList").addEventListener("input", () => {
  if (!S.project) return;
  clearTimeout(_shotTimer);
  _shotTimer = setTimeout(() => {
    api("/api/project/patch", { project: S.project.id, shot_list: $("#sbShotList").value })
      .catch(() => { /* 静默：拍摄清单是可选输入 */ });
  }, 900);
});

$("#btnSbGenerate").addEventListener("click", () => {
  if (!S.project) return toast("Select a project first");
  const prompt = $("#sbPrompt").value.trim();
  if (!prompt) return toast("The prompt cannot be empty");
  const ref = $("#sbRef").value;
  const box = $("#sbJob");
  box.innerHTML = `<div class="job-msg">Seedream image-set job submitted… (4K sets usually take 30–120 s)</div>
    <div class="progress"><div style="width:5%"></div></div>`;
  $("#btnSbGenerate").disabled = true;
  api("/api/storyboard/generate", {
    project: S.project.id, prompt,
    shot_list: $("#sbShotList").value.trim(),
    images: ref ? [ref] : undefined,
    size: $("#sbSize").value, max_images: +$("#sbCount").value,
    output_format: $("#sbFmt").value,
    optimize_prompt: ((S.config.storyboard_api || {}).optimize_prompt || ""),
  }).then(({ job }) => pollJob(job, box, async (res) => {
    $("#btnSbGenerate").disabled = false;
    box.innerHTML = `<div class="job-msg">Generation complete ✓ ${res.count} image(s) · ${res.model} · ${res.size} · ${res.elapsed}s${res.note ? " · " + res.note : ""}</div>`;
    const { project } = await api("/api/project?id=" + S.project.id);
    S.project = project;
    renderStoryboards(); renderFirstFrame();
    toast(`${res.count} handheld storyboards generated`);
  })).catch(e => {
    $("#btnSbGenerate").disabled = false;
    box.innerHTML = `<div class="job-error">✕ ${e.message}</div>`;
  });
});

/* ================= 步骤5 合成导出 ================= */
// 汇总当前项目的全部候选段（基础运镜 + 手持展示）
function collectSegs() {
  const segs = [];
  (S.project.cameramove || []).forEach((v, i) =>
    segs.push({ ...v, kind: "cameramove", key: "cm:" + v.file, label: v.name || ("Camera move " + (i + 1)) }));
  (S.project.handheld_videos || []).forEach(v => {
    if (v.kind === "showcase_seg" || v.kind === "showcase_hero") {
      segs.push({ ...v, kind: "handheld", key: "hh:" + v.file,
        label: v.name || v.file, sc: true, order: v.order });
    } else {
      segs.push({ ...v, kind: "handheld", key: "hh:" + v.file, label: "Handheld " + v.file });
    }
  });
  // 分段展示段按产品顺序排在前
  segs.sort((a, b) => (b.sc ? 1 : 0) - (a.sc ? 1 : 0) ||
    ((a.order || 50) - (b.order || 50)) ||
    String(a.created || "").localeCompare(String(b.created || "")));
  return segs;
}

// 默认勾选：基础运镜只取最新一段，手持全部勾选
function defaultSegSelection(segs) {
  const cmSegs = segs.filter(s => s.kind === "cameramove");
  const latestCm = cmSegs.length ? cmSegs[cmSegs.length - 1].key : null;
  return new Set(segs.filter(s => s.kind === "handheld" || s.key === latestCm).map(s => s.key));
}

function renderSegListBox() {
  // 项目切换守卫：换项目时重置顺序与勾选（避免把上个项目的选择带过来）
  const pid = (S.project || {}).id || null;
  if (S.segSelPid !== pid) {
    S.segSelPid = pid;
    S.segOrder = [];
    S.segSel = null;
  }
  const segs = collectSegs();
  // 保持已有顺序
  if (S.segOrder.length) {
    segs.sort((a, b) => (S.segOrder.indexOf(a.file) + 1 || 99) - (S.segOrder.indexOf(b.file) + 1 || 99));
  }
  S.segOrder = segs.map(s => s.file);
  if (!S.segSel) S.segSel = defaultSegSelection(segs);
  // 清理已不存在的勾选
  const keys = new Set(segs.map(s => s.key));
  S.segSel = new Set([...S.segSel].filter(k => keys.has(k)));
  const box = $("#segList");
  box.innerHTML = segs.length ? segs.map((s, i) => `
    <div class="seg ${S.segSel.has(s.key) ? "on" : ""}">
      <label class="seg-pick"><input type="checkbox" ${S.segSel.has(s.key) ? "checked" : ""}
        onchange="toggleSeg('${encodeURIComponent(s.key)}', this.checked)"></label>
      <span class="idx">${i + 1}</span>
      <span class="name">${s.label}${s.kind === "cameramove" ? ' <span class="tag">Camera move</span>' : ' <span class="tag">Handheld</span>'}</span>
      <span class="dur">${fmtDur(s.dur)}</span>
      <button class="btn" onclick="moveSeg(${i},-1)" ${i === 0 ? "disabled" : ""}>↑</button>
      <button class="btn" onclick="moveSeg(${i},1)" ${i === segs.length - 1 ? "disabled" : ""}>↓</button>
    </div>`).join("") : `<div class="empty">No video segments yet — produce a camera-move film or an AI handheld video in step 3 first</div>`;
}

window.toggleSeg = (keyEnc, on) => {
  const key = decodeURIComponent(keyEnc);
  if (on) S.segSel.add(key); else S.segSel.delete(key);
  document.querySelectorAll("#segList .seg").forEach(el => {
    const cb = el.querySelector('input[type="checkbox"]');
    if (cb) el.classList.toggle("on", cb.checked);
  });
};

async function renderCompose() {
  renderSegListBox();
  // BGM
  if (!S.bgmTracks.length) {
    const { tracks } = await api("/api/bgm");
    S.bgmTracks = tracks;
    $("#bgmList").innerHTML = tracks.map((t, i) => `
      <div class="bgm-card ${i === 0 ? "sel" : ""}" data-i="${i}">
        <b>♪ ${t.name}</b><span class="lic">${t.license}</span>
        <audio controls preload="none" src="/bgm/${encodeURIComponent(t.name + ".mp3")}"></audio>
      </div>`).join("");
    $$(".bgm-card").forEach(c => c.addEventListener("click", () => {
      $$(".bgm-card").forEach(x => x.classList.remove("sel"));
      c.classList.add("sel");
    }));
  }
}
window.moveSeg = (i, d) => {
  const arr = S.segOrder;
  const j = i + d;
  if (j < 0 || j >= arr.length) return;
  [arr[i], arr[j]] = [arr[j], arr[i]];
  S.segOrder = arr;
  renderSegListBox();
};
$("#bgmGain").addEventListener("input", () => $("#bgmGainVal").textContent = $("#bgmGain").value + " dB");

$("#btnCompose").addEventListener("click", () => {
  const segs = collectSegs();
  if (!segs.length) return toast("No video segments to compose");
  // 按显示顺序，只取勾选段
  segs.sort((a, b) => (S.segOrder.indexOf(a.file) + 1 || 99) - (S.segOrder.indexOf(b.file) + 1 || 99));
  const picked = segs.filter(s => S.segSel.has(s.key));
  if (!picked.length) return toast("Select at least one video segment first");
  // 优先绝对路径（手持条目的 file 可能是裸文件名，path 才是绝对路径）
  const segments = picked.map(f => f.path || f.file);
  const bgmSel = document.querySelector(".bgm-card.sel");
  const bgm = bgmSel ? S.bgmTracks[+bgmSel.dataset.i].file : null;
  const box = $("#composeJob");
  box.innerHTML = `<div class="job-msg">Composing: stitching ${segments.length} clip(s) + mixing audio…</div>
    <div class="progress"><div style="width:5%"></div></div>`;
  api("/api/compose", {
    project: S.project.id, segments, bgm,
    gain_db: +$("#bgmGain").value,
    fade_in: +$("#fadeIn").value,
    fade_out: +$("#fadeOut").value,
  }).then(({ job }) => pollJob(job, box, (res) => {
    box.innerHTML = `<div class="job-msg">Composition complete ✓</div>`;
    $("#composeOut").innerHTML = `
      <h3>Final Cut</h3>
      <video class="result" style="max-width:320px" controls src="/media/${S.project.id}/${encodeURIComponent(basename(res.file))}"></video>
      <div class="file-path">${res.file}<br>
        <a class="btn" style="margin-top:6px;display:inline-block" href="/media/${S.project.id}/${encodeURIComponent(basename(res.file))}" download>⬇ Download final cut</a>
      </div>`;
    toast("Final cut exported");
  })).catch(e => box.innerHTML = `<div class="job-error">✕ ${e.message}</div>`);
});

/* ---------------- 启动 ---------------- */
async function boot() {
  // 1) 官方指令词（服务端为准，失败则用内置兜底）
  try {
    const { prompts } = await api("/api/prompts");
    if (prompts) Object.assign(P, prompts);
  } catch (e) { /* 离线兜底 */ }
  // 2) 配置（模型能力 / 凭据状态）
  try {
    const { config } = await api("/api/config");
    S.config = config;
    VID_MODELS = config.video_models || [];
    IMG_MODELS = config.image_models || [];
  } catch (e) { /* 忽略 */ }
  // 3) 指令词就位后再写入默认值（此刻用户尚未编辑，不会被覆盖）
  $("#hhPrompt").value = P.video_handheld;
  $("#sbPrompt").value = P.storyboard_handheld;
  // 4) 干净起点：不恢复上次项目/步骤，列表只放本次会话项目
  S.sessionIds = [];
  S.showAll = false;
  await refreshProjects();
  goto(1);
}
boot();
