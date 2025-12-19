# app.py
# Fully Self-contained FastAPI + Template Auto-Creator + Simulation Web UI

import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import json

# =========================================================
# (1) TEMPLATE AUTO-GENERATION (index.html + result.html)
# =========================================================

TEMPLATE_DIR = "templates"
STATIC_DIR = "static"

os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ---------- index.html 생성 ----------
INDEX_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8" />
<title>3D Printing Smart Factory Simulation</title>
<script>
    async function runSimulation() {
        const payload = collectFormData();
        const res = await fetch("/run-simulation", {
            method: "POST",
            headers: {"Content-Type":"application/json"},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        localStorage.setItem("sim_result", JSON.stringify(data));
        window.location.href = "/result";
    }

    function collectFormData() {
        return {
            demand: {
                sim_duration_min: parseInt(document.getElementById("sim_duration").value),
                order_cycle_min: parseInt(document.getElementById("order_cycle").value),
                patients_min: parseInt(document.getElementById("patients_min").value),
                patients_max: parseInt(document.getElementById("patients_max").value),
                items_min: parseInt(document.getElementById("items_min").value),
                items_max: parseInt(document.getElementById("items_max").value),
                order_due_date: parseInt(document.getElementById("order_due").value || "0")
            },
            material: {
                part_weight_g: parseFloat(document.getElementById("mat_weight").value || "0"),
                resin_cost_per_kg: parseFloat(document.getElementById("mat_resin_cost").value || "0"),
                pallet_size_limit: parseInt(document.getElementById("mat_pallet_limit").value || "0"),
                initial_platforms: parseInt(document.getElementById("mat_initial_plat").value || "0")
            },
            printing: {
                printer_count: parseInt(document.getElementById("print_cnt").value || "1"),
                print_time_min: parseFloat(document.getElementById("print_time").value || "0"),
                defect_rate: parseFloat(document.getElementById("print_defect").value || "0"),
                breakdown_enabled: document.getElementById("print_breakdown") ? document.getElementById("print_breakdown").checked : false,
                mtbf_min: parseFloat(document.getElementById("print_mtbf").value || "0"),
                mttr_min: parseFloat(document.getElementById("print_mttr").value || "0"),
                maintenance_cycle_h: parseFloat(document.getElementById("print_maint_cycle").value || "0"),
                maintenance_duration_min: parseFloat(document.getElementById("print_maint_dur").value || "0")
            },
            preprocess: {
                healing_time: parseFloat(document.getElementById("pre_heal").value || "0"),
                placement_time: parseFloat(document.getElementById("pre_place").value || "0"),
                support_time: parseFloat(document.getElementById("pre_support").value || "0"),
                transfer_mode: document.getElementById("pre_transfer") ? document.getElementById("pre_transfer").value : "amr"
            },
            platform_clean: {
                washer_count: parseInt(document.getElementById("plat_washer_cnt").value || "0"),
                platform_clean_time: parseFloat(document.getElementById("plat_clean_time").value || "0")
            },
            auto_post_common: {
                wash1_count: parseInt(document.getElementById("w1_cnt").value || "0"),
                wash1_time: parseFloat(document.getElementById("w1_time").value || "0"),
                wash2_count: parseInt(document.getElementById("w2_cnt").value || "0"),
                wash2_time: parseFloat(document.getElementById("w2_time").value || "0"),
                dry_count: parseInt(document.getElementById("dry_cnt").value || "0"),
                dry_time: parseFloat(document.getElementById("dry_time").value || "0"),
                uv_count: parseInt(document.getElementById("uv_cnt").value || "0"),
                uv_time: parseFloat(document.getElementById("uv_time").value || "0"),
                defect_wash1: parseFloat(document.getElementById("def_w1").value || "0"),
                defect_wash2: parseFloat(document.getElementById("def_w2").value || "0"),
                defect_dry: parseFloat(document.getElementById("def_dry").value || "0"),
                defect_uv: parseFloat(document.getElementById("def_uv").value || "0")
            },
            auto_post_amr: {
                amr_count: parseInt(document.getElementById("amr_cnt").value || "0"),
                amr_speed: parseFloat(document.getElementById("amr_speed").value || "0"),
                load_time: parseFloat(document.getElementById("amr_load").value || "0"),
                unload_time: parseFloat(document.getElementById("amr_unload").value || "0"),
                dist_printer_w1: parseFloat(document.getElementById("dist_pw1").value || "0"),
                dist_w1_w2: parseFloat(document.getElementById("dist_w1w2").value || "0"),
                dist_w2_dry: parseFloat(document.getElementById("dist_w2d").value || "0"),
                dist_dry_uv: parseFloat(document.getElementById("dist_duv").value || "0"),
                shift_enabled: document.getElementById("amr_shift") ? document.getElementById("amr_shift").checked : false
            },
            manual_transport: {
                worker_count: parseInt(document.getElementById("man_cnt").value || "0"),
                walk_speed: parseFloat(document.getElementById("man_speed").value || "0"),
                dist_printer_w1: parseFloat(document.getElementById("man_pw1").value || "0"),
                dist_w1_w2: parseFloat(document.getElementById("man_w1w2").value || "0"),
                dist_w2_dry: parseFloat(document.getElementById("man_w2d").value || "0"),
                dist_dry_uv: parseFloat(document.getElementById("man_duv").value || "0"),
                shift_start: (document.getElementById("man_shift_s") || {}).value || "09:00",
                shift_end: (document.getElementById("man_shift_e") || {}).value || "18:00",
                workdays: parseInt((document.getElementById("man_workdays") || {value: "7"}).value || "7")
            },
            manual_post: {
                support_remove_time: parseFloat(document.getElementById("mp_sup").value || "0"),
                finish_time: parseFloat(document.getElementById("mp_finish").value || "0"),
                paint_time: parseFloat(document.getElementById("mp_paint").value || "0"),
                move_time: parseFloat(document.getElementById("mp_move").value || "0")
            },
            stacker: {
                enabled: document.getElementById("stk_enabled") ? document.getElementById("stk_enabled").checked : true,
                max_wip: parseInt(document.getElementById("stk_wip").value || "0"),
                order_policy: document.getElementById("stk_policy") ? document.getElementById("stk_policy").value : "FIFO"
            },
            cost: {
                labor_cost_hour: parseFloat(document.getElementById("cost_labor").value || "0"),
                printer_price: parseFloat(document.getElementById("cost_printer").value || "0"),
                washer_price: parseFloat(document.getElementById("cost_washer").value || "0"),
                dryer_price: parseFloat(document.getElementById("cost_dryer").value || "0"),
                uv_price: parseFloat(document.getElementById("cost_uv").value || "0"),
                amr_price: parseFloat(document.getElementById("cost_amr").value || "0"),
                depreciation_years: parseInt(document.getElementById("cost_dep").value || "5"),
                overhead_month: parseFloat(document.getElementById("cost_over").value || "0")
            }
        };
    }
</script>

<style>
body { font-family: Pretendard, sans-serif; margin:20px; }
.section { border:1px solid #ccc; padding:15px; margin-bottom:20px; border-radius:6px; }
.row { display:flex; gap:10px; margin-bottom:8px; }
.row label { width:240px; }
input,select { padding:4px; }
button { padding:10px 20px; background:#0066ff; color:white; border:none; border-radius:5px; cursor:pointer; }
</style>
</head>

<body>
<h1>3D Printing Smart Factory – Simulation Config</h1>

<div class="section">
<h2>테스트 예시 입력 (최소셋)</h2>
<div class="row">
    <label>시뮬레이션 기간(분):</label>
    <input id="sim_duration" value="4320">
</div>
<div class="row">
    <label>주문주기(분):</label>
    <input id="order_cycle" value="120">
</div>
<div class="row">
    <label>환자 범위 (min ~ max):</label>
    <input id="patients_min" value="1"> ~ <input id="patients_max" value="3">
</div>
<div class="row">
    <label>아이템 범위 (min ~ max):</label>
    <input id="items_min" value="1"> ~ <input id="items_max" value="3">
</div>
</div>

<button onclick="runSimulation()">시뮬레이션 실행</button>

</body>
</html>
"""

# ---------- result.html 생성 ----------
RESULT_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Simulation Result</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<style>
body { font-family: Pretendard, sans-serif; margin:20px; }
.box { border:1px solid #ccc; padding:15px; margin-bottom:25px; border-radius:6px; }
.chart { width:100%; height:450px; }
pre { background:#f4f4f4; padding:10px; max-height:300px; overflow:auto; }
.notice { color:#666; font-size:13px; }
</style>
</head>

<body>

<h1> 시뮬레이션 결과</h1>
<button onclick="window.location.href='/'">← 설정 페이지로</button>

<hr>

<div class="box">
  <h2>1) KPI Summary</h2>
  <div id="kpi"></div>
</div>

<div class="box">
  <h2>2) Stacker WIP 변화 그래프</h2>
  <div id="stacker_wip_chart" class="chart"></div>
</div>

<div class="box">
  <h2>3) AMR 동선 Heatmap / Count</h2>
  <div id="amr_heatmap" class="chart"></div>
</div>

<div class="box">
  <h2>4) Scrap 발생 위치 / 빈도</h2>
  <div id="scrap_chart" class="chart"></div>
</div>

<div class="box">
  <h2>5) Raw JSON (디버깅용)</h2>
  <pre id="raw_json"></pre>
</div>

<div class="box">
  <h2>6) Factory Gantt Chart (trace_events 기반)</h2>
  <div id="gantt_chart" class="chart"></div>
  <div class="notice">
    result.trace_events 에 기록된 Printer / Wash / Dry / UV / AMR / Worker 사용시간을 자원별 타임라인으로 시각화합니다.
  </div>
</div>

<script>
const sim = JSON.parse(localStorage.getItem("sim_result") || "null");

if(!sim){
  document.getElementById("raw_json").textContent = "localStorage.sim_result 가 없습니다. 먼저 시뮬레이션을 실행하세요.";
} else {
  document.getElementById("raw_json").textContent = JSON.stringify(sim, null, 2);
}

const result = sim && sim.result ? sim.result : {};

// ==================== 1) KPI Summary ====================
(function renderKPI(){
  const kpiDiv = document.getElementById("kpi");
  const kpi = result.kpi || result.KPI || {};

  if(!kpi || Object.keys(kpi).length === 0){
    kpiDiv.innerHTML = '<div class="notice">kpi 정보가 result.kpi 에 없습니다. (백엔드에서 kpi dict를 result.kpi로 넘겨주면 여기에 표시됩니다)</div>';
    return;
  }

  let html = "<ul>";
  for(const [k,v] of Object.entries(kpi)){
    if(typeof v === "object") continue; // utilization 같은 nested dict는 여기선 스킵
    html += `<li><b>${k}</b>: ${v}</li>`;
  }
  html += "</ul>";
  kpiDiv.innerHTML = html;
})();

// ==================== 2) Stacker WIP Graph ====================
(function renderStackerWIP(){
  const div = document.getElementById("stacker_wip_chart");
  const raw = result.stacker_wip_history || result.stackerWipHistory || null;

  if(!raw || !Array.isArray(raw) || raw.length === 0){
    div.innerHTML = '<div class="notice">stacker_wip_history 데이터가 없습니다. (result.stacker_wip_history 에 [ {t, wip}, ... ] 형식으로 넘겨주세요)</div>';
    return;
  }

  const t = [];
  const wip = [];
  raw.forEach(d => {
    if(Array.isArray(d)){
      t.push(d[0]);
      wip.push(d[1]);
    }else if(typeof d === "object"){
      t.push(d.t ?? d.time ?? 0);
      wip.push(d.wip ?? d.value ?? 0);
    }
  });

  const trace = {
    x: t,
    y: wip,
    mode: "lines+markers",
    name: "Stacker WIP"
  };

  Plotly.newPlot(div, [trace], {
    title: "Stacker WIP over Time",
    xaxis: { title: "Time (min)" },
    yaxis: { title: "WIP (Platforms/Jobs)" }
  });
})();

// ==================== 3) AMR Route Heatmap / Count ====================
(function renderAMRHeatmap(){
  const div = document.getElementById("amr_heatmap");
  const counts = result.amr_route_counts || result.amrRouteCounts || result.amr_moves || null;

  if(!counts || typeof counts !== "object" || Object.keys(counts).length === 0){
    div.innerHTML = '<div class="notice">AMR 경로 카운트 데이터가 없습니다. (result.amr_route_counts = {"printer_to_wash1":12, ...} 형식으로 넘겨주세요)</div>';
    return;
  }

  // route 명을 from / to 로 나누기 ("printer_to_wash1" → "printer", "wash1")
  const fromSet = new Set();
  const toSet = new Set();
  const routeEntries = [];

  for(const [route, cntRaw] of Object.entries(counts)){
    const cnt = typeof cntRaw === "number" ? cntRaw : parseFloat(cntRaw) || 0;
    let from = route;
    let to = "";
    if(route.includes("_to_")){
      const parts = route.split("_to_");
      from = parts[0];
      to = parts[1];
    }else if(route.includes("->")){
      const parts = route.split("->");
      from = parts[0];
      to = parts[1];
    }
    fromSet.add(from);
    toSet.add(to || "(unknown)");
    routeEntries.push({from, to: to || "(unknown)", cnt});
  }

  const fromList = Array.from(fromSet);
  const toList = Array.from(toSet);

  // 2D matrix [from][to]
  const z = fromList.map(() => toList.map(() => 0));
  routeEntries.forEach(e => {
    const i = fromList.indexOf(e.from);
    const j = toList.indexOf(e.to);
    if(i >= 0 && j >= 0){
      z[i][j] = e.cnt;
    }
  });

  const data = [{
    z: z,
    x: toList,
    y: fromList,
    type: "heatmap",
    colorscale: "Viridis",
    hoverongaps: false
  }];

  Plotly.newPlot(div, data, {
    title: "AMR Route Count Heatmap (from → to)",
    xaxis: { title: "To" },
    yaxis: { title: "From" }
  });
})();

// ==================== 4) Scrap by Stage Bar Chart ====================
(function renderScrap(){
  const div = document.getElementById("scrap_chart");
  const scrap = result.scrap_by_stage || result.scrapByStage || null;

  if(!scrap || typeof scrap !== "object" || Object.keys(scrap).length === 0){
    div.innerHTML = '<div class="notice">Scrap 단계별 데이터가 없습니다. (result.scrap_by_stage = {"Print":10, "WashM1":3, ...} 형식으로 넘겨주세요)</div>';
    return;
  }

  const stages = Object.keys(scrap);
  const counts = stages.map(s => {
    const v = scrap[s];
    return typeof v === "number" ? v : parseFloat(v) || 0;
  });

  const trace = {
    x: stages,
    y: counts,
    type: "bar"
  };

  Plotly.newPlot(div, [trace], {
    title: "Scrap 발생 위치 / 빈도",
    xaxis: { title: "Stage" },
    yaxis: { title: "Scrap Count" }
  });
})();

// ==================== 6) Factory Gantt Chart (trace_events) ====================
(function renderGantt(){
  const div = document.getElementById("gantt_chart");
  if(!div) return;

  const traceEvents = result.trace_events || result.traceEvents || null;

  if(!traceEvents || !Array.isArray(traceEvents) || traceEvents.length === 0){
    div.innerHTML = '<div class="notice">trace_events 데이터가 없습니다. (백엔드에서 result.trace_events = [{Resource, t0, t1, stage, id}, ...] 형식으로 넘겨주세요)</div>';
    return;
  }

  // 자원(Resource) 목록 추출
  const resources = Array.from(
    new Set(
      traceEvents.map(ev => ev.Resource || ev.stage || "Unknown")
    )
  );

  const data = [];

  // 자원(Resource)별로 하나의 trace (수평 bar) 생성
  resources.forEach(res => {
    const evs = traceEvents.filter(ev => (ev.Resource || ev.stage || "Unknown") === res);

    const x = [];      // duration (t1 - t0)
    const base = [];   // 시작 시간 t0
    const y = [];      // y축은 전부 같은 자원 이름
    const text = [];   // hover text

    evs.forEach(ev => {
      const t0 = parseFloat(ev.t0 || 0);
      const t1 = parseFloat(ev.t1 || 0);
      const dur = Math.max(t1 - t0, 0.001); // 0 길이 방지

      x.push(dur);
      base.push(t0);
      y.push(res);
      const jobId = ev.id || ev.job_id || "";
      const stage = ev.stage || "";
      text.push(`${res} | ${stage} | ${jobId} (t=${t0.toFixed(1)}~${t1.toFixed(1)})`);
    });

    data.push({
      type: "bar",
      orientation: "h",
      x: x,
      y: y,
      base: base,
      name: res,
      hovertext: text,
      hoverinfo: "text"
    });
  });

  const layout = {
    title: "Factory Resource Gantt (Timeline)",
    barmode: "stack",
    xaxis: {
      title: "Time (min)",
      rangemode: "nonnegative"
    },
    yaxis: {
      title: "Resource",
      automargin: true
    },
    margin: {l: 120, r: 20, t: 40, b: 40}
  };

  Plotly.newPlot(div, data, layout);
})();
</script>

</body>
</html>
"""

# 파일 자동 생성
with open(os.path.join(TEMPLATE_DIR, "index.html"), "w", encoding="utf-8") as f:
    f.write(INDEX_HTML)

with open(os.path.join(TEMPLATE_DIR, "result.html"), "w", encoding="utf-8") as f:
    f.write(RESULT_HTML)


# =========================================================
# (2) FASTAPI 서버
# =========================================================

app = FastAPI(title="3DPF Web Simulator", version="2.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------- Pydantic 모델 ----------------
class SimulationFullConfig(BaseModel):
    demand: Dict[str, Any]
    material: Dict[str, Any]
    printing: Dict[str, Any]
    preprocess: Dict[str, Any]
    platform_clean: Dict[str, Any]
    auto_post_common: Dict[str, Any]
    auto_post_amr: Dict[str, Any]
    manual_transport: Dict[str, Any]
    manual_post: Dict[str, Any]
    stacker: Dict[str, Any]
    cost: Dict[str, Any]


# ---------------- ROUTES ----------------

@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(TEMPLATE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/result", response_class=HTMLResponse)
async def result_page():
    with open(os.path.join(TEMPLATE_DIR, "result.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.post("/run-simulation")
async def run_simulation(config: SimulationFullConfig):
    """
    Web에서 설정받은 config는 일단 로그로 찍어두고,
    main_SimPy.run_full_simulation()을 호출해서 result에 넣음.
    run_full_simulation은 dict(result) 를 리턴해야
    프론트에서 KPI / trace_events 등을 사용할 수 있음.
    """
    print("\n[WEB] Config Received:")
    print(json.dumps(config.dict(), indent=2, ensure_ascii=False))

    try:
        from main_SimPy import run_full_simulation

        # 프론트에서 받은 시뮬레이션 시간 사용
        sim_time = int(config.demand.get("sim_duration_min", 4320) or 4320)

        # 웹에서는 matplotlib 팝업 안 띄우고 데이터만 반환
        result = run_full_simulation(sim_duration=sim_time, show_gantt=False)

    except Exception as e:
        return {"status": "error", "message": str(e)}

    return {
        "status": "success",
        "config": config.dict(),
        "result": result,
    }


# =========================================================
# (3) 실행부
# =========================================================
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
