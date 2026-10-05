"use strict";

const $ = (id) => document.getElementById(id);
let files = [];

// 快门时间支持 "0.004" 或 "1/250" 两种写法，返回秒（非法返回 NaN）
function parseShutter(str) {
  const s = String(str).trim();
  if (!s) return NaN;
  if (s.includes("/")) {
    const parts = s.split("/");
    if (parts.length !== 2) return NaN;
    const a = parseFloat(parts[0]);
    const b = parseFloat(parts[1]);
    if (!isFinite(a) || !isFinite(b) || b === 0) return NaN;
    return a / b;
  }
  return parseFloat(s);
}

$("files").addEventListener("change", (e) => {
  files = Array.from(e.target.files);
  renderList();
});

function renderList() {
  const box = $("fileList");
  box.innerHTML = "";
  const n = files.length;
  $("countHint").textContent = n
    ? `已选 ${n} 张（需要 3～7 张）`
    : "";
  const defaultRef = Math.floor((n - 1) / 2);
  files.forEach((f, i) => {
    const row = document.createElement("div");
    row.className = "row";

    const img = document.createElement("img");
    img.src = URL.createObjectURL(f);
    img.alt = f.name;

    const name = document.createElement("span");
    name.className = "name";
    name.textContent = f.name;
    name.title = f.name;

    const shutter = document.createElement("input");
    shutter.className = "shutter";
    shutter.placeholder = "快门秒数，如 1/250 或 0.004";
    shutter.addEventListener("input", validate);

    const label = document.createElement("label");
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "ref";
    radio.value = String(i);
    if (i === defaultRef) radio.checked = true;
    label.append(radio, " 参照帧");

    row.append(img, name, shutter, label);
    box.append(row);
  });
  validate();
}

function validate() {
  const n = files.length;
  let ok = n >= 3 && n <= 7;
  document.querySelectorAll("#fileList .shutter").forEach((inp) => {
    const v = parseShutter(inp.value);
    const good = isFinite(v) && v > 0;
    inp.classList.toggle("bad", !good);
    if (!good) ok = false;
  });
  if (!document.querySelector('input[name="ref"]:checked')) ok = false;
  $("go").disabled = !ok;
  return ok;
}

$("go").addEventListener("click", async () => {
  if (!validate()) return;
  $("err").hidden = true;
  $("results").hidden = true;
  $("busy").hidden = false;
  $("go").disabled = true;
  try {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    const shutters = Array.from(
      document.querySelectorAll("#fileList .shutter")
    ).map((inp) => parseShutter(inp.value));
    fd.append("shutters", JSON.stringify(shutters));
    fd.append(
      "reference",
      document.querySelector('input[name="ref"]:checked').value
    );
    const resp = await fetch("/api/merge", { method: "POST", body: fd });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.detail || `合成失败（HTTP ${resp.status}）`);
    }
    showResults(data);
  } catch (e) {
    $("err").textContent = "错误：" + e.message;
    $("err").hidden = false;
  } finally {
    $("busy").hidden = true;
    validate();
  }
});

function showResults(d) {
  const bust = "?t=" + Date.now();
  $("imgResult").src = d.result_url + bust;
  $("imgRef").src = d.reference_url + bust;
  $("imgMask").src = d.mask_url + bust;
  $("dlResult").href = d.result_url;
  $("dlMask").href = d.mask_url;
  $("stats").textContent = JSON.stringify(d.stats, null, 2);
  $("results").hidden = false;
  syncCompare();
  $("results").scrollIntoView({ behavior: "smooth" });
}

// 让上层（参照帧）图片与容器同宽，裁切由 .top 的宽度控制
function syncCompare() {
  const c = $("compare");
  $("imgRef").style.width = c.clientWidth + "px";
}
window.addEventListener("resize", syncCompare);
$("imgResult").addEventListener("load", syncCompare);

$("slider").addEventListener("input", () => {
  $("compareTop").style.width = $("slider").value / 10 + "%";
});
