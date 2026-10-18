import { evaluate as ev, send, waitFor, close, events } from "./cdp.mjs";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
const exec = promisify(execFile),
  out = path.resolve("ui/out");
await mkdir(out, { recursive: true });
const shots = [],
  checks = [];
await send("Runtime.enable");
await ev("performance.clearMarks();performance.clearMeasures()");
await send("Log.enable");
const click = async (text, selector = "button") =>
  ev(
    `(()=>{const e=[...document.querySelectorAll(${JSON.stringify(selector)})].find(e=>e.textContent.trim()===${JSON.stringify(text)});if(!e)throw new Error('Missing '+${JSON.stringify(text)});e.click();return true;})()`,
  );
const clickStarting = async (prefix) =>
  ev(
    `(()=>{const e=[...document.querySelectorAll("button")].find(e=>e.textContent.trim().startsWith(${JSON.stringify(prefix)}));if(!e)throw new Error('Missing '+${JSON.stringify(prefix)});e.click();return true;})()`,
  );
async function shot(name) {
  await ev(
    "new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))",
  );
  await new Promise((r) => setTimeout(r, 500));
  const file = path.join(out, name + ".png");
  await exec("screenshot", [file]);
  shots.push(file);
  console.log("screenshot", name);
}
async function route(hash, condition) {
  await ev(`location.hash=${JSON.stringify(hash)}`);
  await waitFor(condition, 60000);
}
try {
  await route("/runs", '!!document.querySelector(".run-table")');
  await ev(
    '(()=>{const clear=[...document.querySelectorAll(".selection-bar button")].find(e=>e.textContent.trim()==="Clear");if(clear)clear.click();})()',
  );
  await shot("01-runs");
  // the newest completed report and three of its runs (any store with one works)
  const { selectedIds, reportId } = await ev(
    `(async()=>{const t=sessionStorage.getItem('kiseki-token');const get=p=>fetch('/api/'+p,{headers:{'X-Kiseki-Token':t}}).then(r=>r.json());const report=(await get('reports')).filter(r=>r.status.state==='completed').sort((a,b)=>b.id.localeCompare(a.id))[0];if(!report)throw new Error('Build a report first');return {selectedIds:[...new Set(report.spec.models.map(m=>m.split(':')[0]))].slice(0,3),reportId:report.id};})()`,
  );
  for (const id of selectedIds)
    await ev(
      `(()=>{const e=document.querySelector('input[aria-label="Select ${id}"]');if(!e.checked)e.click();})()`,
    );
  await shot("02-runs-selected");
  await route(
    "/runs/" + selectedIds[0],
    'document.querySelectorAll(".uplot").length===2',
  );
  await shot("03-run-detail");
  await click("configuration");
  await shot("04-run-configuration");
  await clickStarting("events");
  await shot("05-run-events");
  await clickStarting("models");
  await shot("06-run-models");
  await route("/new", '!!document.querySelector("form")');
  await shot("07-new-run");
  await route(
    "/compare",
    `document.querySelectorAll(".uplot").length===1&&document.querySelectorAll("section.panel>.table-wrap tbody tr").length===${selectedIds.length}`,
  );
  await waitFor(
    '!document.querySelector("main").innerText.includes("Loading…")',
  );
  await shot("08-compare");
  checks.push({
    comparison: await ev(
      '[...document.querySelectorAll("tbody tr")].map(e=>e.innerText)',
    ),
  });
  await route("/reports", '!!document.querySelector(".report-card")');
  await shot("09-reports");
  await route(
    "/reports/new",
    `document.querySelectorAll(".model-choice").length===${selectedIds.length}`,
  );
  await shot("10-new-report");
  await route(
    "/reports/" + reportId,
    '!!document.querySelector(".report-nav")',
  );
  await waitFor(
    "[...document.images].filter(i=>i.getBoundingClientRect().top<innerHeight).every(i=>i.complete)",
  );
  await shot("11-report-metrics");
  const sections = await ev(
    '[...document.querySelectorAll(".report-nav button")].map(e=>e.textContent)',
  );
  for (const name of sections) {
    if (/embeddings|relevance/i.test(name)) continue;
    await click(name, ".report-nav button");
    await ev('document.querySelector("main").scrollTo(0,0)');
    await waitFor(
      "[...document.images].filter(i=>i.getBoundingClientRect().top<innerHeight).every(i=>i.complete)",
    );
    await shot(
      "report-" +
        name
          .slice(2)
          .trim()
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "-"),
    );
  }
  await click(
    sections.find((n) => /embeddings/i.test(n)),
    ".report-nav button",
  );
  await waitFor('!!document.querySelector(".scatter-wrap canvas")');
  await ev('document.querySelector("main").scrollTo(0,0)');
  await shot("12-embedding-tsne");
  checks.push({
    webgl: await ev(
      '(()=>{const gl=document.querySelector(".scatter-wrap canvas").getContext("webgl");const info=gl.getExtension("WEBGL_debug_renderer_info");return {renderer:info?gl.getParameter(info.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER),points:document.querySelector(".chart-controls").innerText};})()',
    ),
  });
  await click("PCA");
  await waitFor('!!document.querySelector(".scatter-wrap canvas")');
  await shot("13-embedding-pca");
  await click(
    sections.find((n) => /relevance/i.test(n)),
    ".report-nav button",
  );
  await waitFor(
    'document.querySelectorAll(".relevance-grid canvas").length===4',
  );
  await ev('document.querySelector("main").scrollTo(0,0)');
  await shot("14-lrp-gallery");
  await click("Class averages");
  await waitFor(
    'document.querySelectorAll(".relevance-grid canvas").length===3',
  );
  await shot("15-lrp-classes");
  await route(
    "/machines",
    'document.querySelectorAll(".machine-card").length===3',
  );
  await shot("16-machines");
  const timings = await ev(
    'performance.getEntriesByType("measure").filter(e=>e.name.startsWith("kiseki:")).map(e=>({name:e.name,ms:Math.round(e.duration*10)/10}))',
  );
  const resource = await ev(
    'performance.getEntriesByType("resource").filter(e=>e.name.includes("/api/")&&!e.name.includes("events?")).map(e=>({path:e.name.split("/api/")[1].split("?")[0],ms:Math.round(e.duration*10)/10,bytes:e.decodedBodySize}))',
  );
  const errors = events.filter(
    (e) =>
      e.method === "Runtime.exceptionThrown" ||
      (e.method === "Log.entryAdded" && e.params.entry.level === "error"),
  );
  const result = {
    date: new Date().toISOString(),
    shots,
    checks,
    timings,
    resource,
    errors,
  };
  await writeFile(
    path.join(out, "verification.json"),
    JSON.stringify(result, null, 2),
  );
  console.log(JSON.stringify({ timings, errors }, null, 2));
  if (errors.length) process.exitCode = 1;
} finally {
  close();
}
