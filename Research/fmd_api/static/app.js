"use strict";
const $ = id => document.getElementById(id);
const modelInfo = [
  {id:"baseline", label:"Baseline", description:"원본 데이터 학습 · 증강 없음", color:"#657a8a"},
  {id:"magnitude_warping", label:"Magnitude warping", description:"변환형 증강 1,000개", color:"#c99035"},
  {id:"fmd", label:"FMD", description:"Filtered Masking Diffusion 증강 750개", color:"#0b8178"}
];
const state = {models:null, sample:null, catalog:null, history:[], last:null, busy:false, replaying:false, stop:false};
const escapeText = text => String(text).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = (value, digits=1) => Number.isFinite(value) ? value.toFixed(digits) : "—";
function showError(error) { $("global-error").textContent = error ? error.message || String(error) : ""; $("global-error").hidden = !error; }
async function request(path, options={}) {
  let response;
  try { response = await fetch(path, {cache:"no-store", ...options}); }
  catch { throw new Error("서버에 연결할 수 없습니다. API 서버가 실행 중인지 확인해주세요."); }
  const data = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map(e => `${e.loc.join(".")}: ${e.msg}`).join(" / ") : typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    throw new Error(`요청 실패 (${response.status}) · ${detail}`);
  }
  return data;
}
function setBusy(busy) {
  state.busy = busy;
  document.querySelectorAll("[data-mutate]").forEach(element => element.disabled = busy);
  $("download-response").disabled = busy || !state.last;
}
function clearPredictions() {
  state.last = null;
  $("round-trip").textContent = "응답 대기";
  $("request-time").textContent = "선택한 입력으로 비교를 실행해주세요.";
  $("response-json").textContent = "비교를 실행하면 서버의 실제 JSON 응답이 표시됩니다.";
  $("download-response").disabled = true;
  renderCards();
}
function renderCards(predictions=null) {
  $("prediction-cards").innerHTML = modelInfo.map(info => {
    const result = predictions?.[info.id];
    const threshold = result?.threshold ?? state.models?.models[info.id]?.threshold;
    const status = result ? (result.prediction === 1 ? "이상" : "정상") : "판정 대기";
    const style = result ? (result.prediction === 1 ? "bad" : "") : "muted";
    return `<article class="prediction-card" style="--accent:${info.color}"><div class="card-head"><h3>${info.label}</h3><span class="pill ${style}">${status}</span></div><p class="description">${info.description}</p><div class="score-row"><strong class="score">${fmt(result?.anomaly_score,3)}</strong><span>이상 점수</span><span class="time">추론 ${fmt(result?.inference_ms)} ms</span></div><div class="bar-track"><div class="bar-fill" style="width:${(result?.anomaly_score || 0)*100}%"></div><span class="threshold-marker" style="left:${(threshold || 0)*100}%"></span></div><div class="bar-legend"><span>0 · 정상 쪽</span><span>기준 ${fmt(threshold,2)}</span><span>1 · 이상 쪽</span></div></article>`;
  }).join("");
}
function validateInput() {
  let payload;
  try { payload = JSON.parse($("request-json").value); } catch { throw new Error("요청 JSON 형식이 올바르지 않습니다. 정상·이상 샘플을 다시 불러오거나 JSON 문법을 확인해주세요."); }
  const features = state.models?.feature_names;
  if (!features || JSON.stringify(payload.feature_names) !== JSON.stringify(features)) throw new Error("feature_names는 RealPower, GateOnTime, SetPower, Speed, Length 순서여야 합니다.");
  if (!Array.isArray(payload.window) || payload.window.length !== 30 || payload.window.some(row => !Array.isArray(row) || row.length !== 5 || row.some(value => typeof value !== "number" || !Number.isFinite(value)))) throw new Error("window에는 유효한 숫자 5개씩, 정확히 30개 행이 필요합니다. null·NaN·무한대는 사용할 수 없습니다.");
  if (Object.keys(payload).some(key => !["feature_names","window"].includes(key))) throw new Error("요청에는 feature_names와 window만 넣어주세요. 정답 label은 입력에서 제외합니다.");
  return payload;
}
function renderSensor() {
  const svg = $("sensor-chart");
  let payload;
  try { payload = validateInput(); } catch { svg.innerHTML = ""; $("sensor-range").textContent = "유효한 JSON을 입력해주세요"; return; }
  const values = payload.window.map(row => row[Number($("feature-select").value)]);
  const min = Math.min(...values), max = Math.max(...values), range = max-min;
  const points = values.map((value,i) => `${12+i*576/29},${range ? 130-(value-min)*108/range : 76}`).join(" ");
  svg.innerHTML = `<path d="M12 22H588M12 76H588M12 130H588" stroke="#e8eff1" stroke-dasharray="4 5" fill="none"/><polygon points="12,140 ${points} 588,140" fill="#e9f5f2"/><polyline points="${points}" stroke="#248f85" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>`;
  $("sensor-range").textContent = `${fmt(min,1)} ~ ${fmt(max,1)}`;
}
async function loadSample(id) {
  const sample = await request(`/dashboard/sample/${id}`);
  state.sample = sample;
  $("request-json").value = JSON.stringify(sample.request, null, 2);
  $("truth").textContent = sample.metadata.label ? "원본 정답 · 이상" : "원본 정답 · 정상";
  $("truth").className = "pill " + (sample.metadata.label ? "bad" : "");
  $("sample-number").textContent = `윈도 #${sample.id+1}`;
  $("sample-source").textContent = sample.metadata.source_file;
  $("sample-time").textContent = `${sample.metadata.start_time} → ${sample.metadata.end_time}`;
  const select = $("sample-select");
  select.querySelectorAll("option[data-temporary]").forEach(option => option.remove());
  if (![...select.options].some(option => Number(option.value) === sample.id)) {
    const option = new Option(`재생 중 · ${sample.metadata.label ? "이상" : "정상"} · 윈도 #${sample.id+1}`, String(sample.id));
    option.dataset.temporary = "true"; select.add(option);
  }
  select.value = String(sample.id);
  clearPredictions(); renderSensor(); showError(null);
}
async function chooseSample(id) {
  if (state.busy) return;
  setBusy(true);
  try { await loadSample(id); $("run-state").textContent = "샘플을 불러왔습니다. 세 모델 비교 실행을 눌러주세요."; }
  catch(error) { showError(error); }
  finally { setBusy(false); }
}
function markCustom() {
  state.sample = null;
  $("truth").textContent = "직접 입력 · 정답 정보 없음"; $("truth").className = "pill muted";
  $("sample-number").textContent = "사용자 JSON"; $("sample-source").textContent = "직접 입력한 센서 데이터";
  $("sample-time").textContent = "시간 순서와 센서 단위를 확인해주세요.";
  clearPredictions(); renderSensor(); $("run-state").textContent = "수정한 JSON으로 API를 호출합니다.";
}
async function executeComparison() {
  const payload = validateInput();
  const sample = state.sample;
  const start = performance.now();
  const response = await request("/compare", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
  const roundTrip = performance.now()-start;
  const now = new Date();
  state.last = {executed_utc:now.toISOString(), http_round_trip_ms:roundTrip, metadata:sample?.metadata || null, request:payload, response};
  renderCards(response.predictions);
  $("round-trip").textContent = `HTTP 왕복 ${fmt(roundTrip)} ms`;
  $("request-time").textContent = `${now.toLocaleTimeString("ko-KR")} 실행 · POST /compare · 200 OK`;
  $("response-json").textContent = JSON.stringify(response, null, 2);
  state.history.unshift({time:now.toLocaleTimeString("ko-KR"), input:sample ? `윈도 #${sample.id+1}` : "직접 입력", truth:sample ? (sample.metadata.label ? "이상" : "정상") : "—", response, roundTrip});
  state.history = state.history.slice(0,10);
  $("history").innerHTML = state.history.map(row => `<tr><td>${escapeText(row.time)}</td><td>${escapeText(row.input)}</td><td>${row.truth}</td>${modelInfo.map(info => `<td><span class="pill ${row.response.predictions[info.id].prediction ? "bad" : ""}">${row.response.predictions[info.id].prediction ? "이상" : "정상"}</span></td>`).join("")}<td>${fmt(row.roundTrip)} ms</td></tr>`).join("");
}
async function compareOnce() {
  if (state.busy) return;
  showError(null); clearPredictions(); setBusy(true); $("run-state").textContent = "원시 센서값을 FastAPI로 보내는 중…";
  try { await executeComparison(); $("run-state").textContent = "완료 · 세 모델의 실제 API 응답을 받았습니다."; }
  catch(error) { showError(error); $("run-state").textContent = "요청에 실패했습니다. 입력 또는 서버 상태를 확인해주세요."; }
  finally { setBusy(false); }
}
async function replay() {
  if (state.busy) return;
  if (!state.sample) { showError(new Error("순차 재생은 원본 샘플을 먼저 선택해야 합니다.")); return; }
  const ids = [...state.sample.next_ids];
  setBusy(true); state.replaying=true; state.stop=false; $("stop-button").hidden=false; showError(null);
  let completed=0;
  try {
    for (const id of ids) {
      if (state.stop) break;
      await loadSample(id); $("run-state").textContent=`저장 데이터 재생 ${completed+1}/${ids.length} · 실제 HTTP 요청 중`;
      await executeComparison(); completed++;
      if (completed < ids.length && !state.stop) await new Promise(resolve => setTimeout(resolve,1000));
    }
    $("run-state").textContent = `${state.stop ? "재생 중지" : "재생 완료"} · ${completed}개 윈도 처리`;
  } catch(error) { showError(error); $("run-state").textContent=`재생 오류 · ${completed}개 윈도 처리 후 중단`; }
  finally { state.replaying=false; $("stop-button").hidden=true; setBusy(false); }
}
function showResults(result) {
  $("window-count").textContent = result.replay.requests.toLocaleString("ko-KR");
  const total = result.replay.requests * modelInfo.length;
  const mismatches = modelInfo.reduce((sum,info) => sum + result.models[info.id].prediction_mismatches,0);
  $("agreement").textContent = `${fmt((total-mismatches)/total*100, mismatches ? 2 : 0)}%`;
  $("record-date").textContent = new Date(result.created_utc).toLocaleString("ko-KR") + " 검증";
  $("metrics-table").innerHTML = modelInfo.map(info => {
    const metrics = result.models[info.id].metrics;
    return `<tr><td><strong>${info.label}</strong></td><td>${state.models.models[info.id].training.generated_anomaly_windows.toLocaleString()}</td>${["precision","recall","f1"].map(key=>`<td>${fmt(metrics[key],4)}</td>`).join("")}<td>${metrics.fn}</td><td>${metrics.fp}</td><td>${fmt(result.single_model_http[info.id].http_round_trip_ms.mean)} ms</td></tr>`;
  }).join("");
  $("benchmark-note").textContent = `세 모델 통합 HTTP: 평균 ${fmt(result.replay.http_round_trip_ms.mean)} ms · p95 ${fmt(result.replay.http_round_trip_ms.p95)} ms · 요청 오류 ${result.replay.errors}건. 별도 초당 ${result.paced_load.target_requests_per_second}회 / ${result.paced_load.requests}회 시험에서 ${result.paced_load.deadline_ms} ms 주기 초과 ${result.paced_load.deadline_misses}건. 이 수치는 저장된 검증 실행 당시의 기록입니다.`;
}
for (const name of ["live","results"]) $("tab-"+name).addEventListener("click", () => {
  for (const other of ["live","results"]) { $("tab-"+other).classList.toggle("selected",other===name); $("tab-"+other).setAttribute("aria-selected",String(other===name)); $("panel-"+other).hidden=other!==name; }
});
$("sample-select").addEventListener("change", event => chooseSample(Number(event.target.value)));
$("normal-sample").addEventListener("click", () => chooseSample(state.catalog.samples.find(sample=>sample.label===0).id));
$("anomaly-sample").addEventListener("click", () => chooseSample(state.catalog.samples.find(sample=>sample.label===1).id));
$("feature-select").addEventListener("change",renderSensor);
$("request-json").addEventListener("input",markCustom);
$("compare-button").addEventListener("click",compareOnce);
$("replay-button").addEventListener("click",replay);
$("stop-button").addEventListener("click",()=>{state.stop=true;$("run-state").textContent="현재 요청이 끝나면 재생을 중지합니다.";});
$("upload-button").addEventListener("click",()=>$("file-input").click());
$("file-input").addEventListener("change",async event=>{
  const file=event.target.files[0]; if(!file) return;
  if(file.size>1024*1024){showError(new Error("요청 JSON은 1MB 이하 파일을 사용해주세요."));return;}
  try {const text=await file.text();$("request-json").value=text;markCustom();validateInput();showError(null);}catch(error){showError(error);}finally{event.target.value="";}
});
$("download-response").addEventListener("click",()=>{
  if(!state.last)return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(state.last,null,2)],{type:"application/json"}));
  const link=document.createElement("a");link.href=url;link.download="weld-api-comparison.json";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
async function init() {
  setBusy(true);renderCards();
  try {
    const [health,models,catalog]=await Promise.all([request("/health"),request("/models"),request("/dashboard/catalog")]);
    state.models=models;state.catalog=catalog;
    $("connection").textContent=`● 서버 연결됨 · ${health.models.length}개 모델`;$("connection").className="pill";
    $("model-count").textContent=String(health.models.length);
    $("sample-select").replaceChildren(...catalog.samples.map(sample=>new Option(`${sample.label ? "이상" : "정상"} · 윈도 #${sample.id+1} · ${sample.source_file}`,String(sample.id))));
    await loadSample(catalog.samples[0].id);
    try {showResults(await request("/dashboard/results"));} catch(error) {$("record-date").textContent="검증 기록을 표시할 수 없습니다.";$("benchmark-note").textContent=error.message;}
    setBusy(false);
  } catch(error) {$("connection").textContent="서버 연결 확인 필요";$("connection").className="pill bad";showError(error);}
}
init();
