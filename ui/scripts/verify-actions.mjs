// Mutating smoke checks: run ONLY against the scratch-store server named in README.
import { evaluate as ev, send, waitFor, close, events } from "./cdp.mjs";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { writeFile } from "node:fs/promises";
import path from "node:path";
const exec = promisify(execFile),
  results = {};
const label = `ui-smoke-${Date.now()}`;
const click = async (text, selector = "button") =>
  ev(
    `(()=>{const e=[...document.querySelectorAll(${JSON.stringify(selector)})].find(e=>e.textContent.trim()===${JSON.stringify(text)});if(!e)throw new Error('Missing '+${JSON.stringify(text)});e.click();})()`,
  );
const field = async (label, value, tag = "input") =>
  ev(
    `(()=>{const e=[...document.querySelectorAll('label.field')].find(e=>e.querySelector('span')?.textContent.toLowerCase()===${JSON.stringify(label.toLowerCase())})?.querySelector(${JSON.stringify(tag)});if(!e)throw new Error('Missing field '+${JSON.stringify(label)});Object.getOwnPropertyDescriptor(${tag === "select" ? "HTMLSelectElement" : "HTMLInputElement"}.prototype,'value').set.call(e,${JSON.stringify(value)});e.dispatchEvent(new Event(${JSON.stringify(tag === "select" ? "change" : "input")},{bubbles:true}));})()`,
  );
const backend = (expression) =>
  ev(
    `(async()=>{const get=async(path,body)=>{const r=await fetch('/api/'+path,{method:body?'POST':'GET',headers:{'X-Kiseki-Token':sessionStorage.getItem('kiseki-token'),'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});if(!r.ok)throw new Error(await r.text());return r.json();};${expression}})()`,
  );
const shot = async (name) => {
  await ev(
    "new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))",
  );
  await new Promise((r) => setTimeout(r, 500));
  await exec("screenshot", [path.resolve("ui/out/" + name + ".png")]);
};
try {
  await send("Runtime.enable");
  await ev('location.hash="/new"');
  await waitFor('!!document.querySelector("form")');
  await field("optimizer", "sgd", "select");
  await field("preset", "smoke", "select");
  await field("seed", "93, 94");
  await field("label", label);
  await waitFor(
    'document.querySelector(".large-number").textContent.includes("2")',
  );
  await shot("17-smoke-grid-form");
  await click("Queue 2 runs →");
  await waitFor('!!document.querySelector(".run-table")');
  const runs = await backend(
    `return (await get('runs')).filter(r=>r.spec.label===${JSON.stringify(label)});`,
  );
  results.runs = runs.map((r) => ({
    id: r.id,
    seed: r.spec.seed,
    machine: r.status.machine,
    iterations: r.spec.iterations,
  }));
  if (
    runs.length !== 2 ||
    runs.some((r) => r.status.machine !== "local" || r.spec.iterations !== 20)
  )
    throw new Error("Smoke form produced the wrong grid");
  await ev(`location.hash=${JSON.stringify("/runs/" + runs.at(-1).id)}`);
  await waitFor('!!document.querySelector(".page-header .badge")');
  const pause = await ev(
    '[...document.querySelectorAll("button")].some(e=>e.textContent.trim()==="Ⅱ Pause")',
  );
  if (pause) {
    await click("Ⅱ Pause");
    await waitFor(
      'document.querySelector(".page-header .badge")?.textContent.trim()==="paused"',
      30000,
    );
    await shot("18-smoke-paused");
    await click("▶ Resume");
    await waitFor(
      '["queued","running","completed"].includes(document.querySelector(".page-header .badge")?.textContent.trim())',
    );
    results.pauseResume = true;
    const canStop = await ev(
      '[...document.querySelectorAll(".page-header button")].some(e=>e.textContent.trim()==="Stop")',
    );
    if (canStop) {
      await click("Stop");
      await waitFor(
        '["stopped","completed"].includes(document.querySelector(".page-header .badge")?.textContent.trim())',
        30000,
      );
      results.stop = await ev(
        'document.querySelector(".page-header .badge").textContent.trim()',
      );
      if (results.stop === "stopped") {
        await shot("18b-smoke-stopped");
        await click("▶ Resume");
      }
    }
  } else results.pauseResume = "Run completed before pause could be requested";
  await waitFor(
    'document.querySelector(".page-header .badge")?.textContent.trim()==="completed"',
    45000,
  );
  await shot("19-smoke-completed");
  for (let attempt = 0; attempt < 60; attempt++) {
    const statuses = await backend(
      `return (await get('runs')).filter(r=>r.spec.label===${JSON.stringify(label)}).map(r=>r.status.state);`,
    );
    if (statuses.every((state) => state === "completed")) break;
    if (statuses.some((state) => state === "failed"))
      throw new Error("Smoke worker failed");
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  results.finalRuns = await backend(
    `return Promise.all(${JSON.stringify(runs.map((r) => r.id))}.map(id=>get('runs/'+id)));`,
  );
  // Create a lightweight metrics report through the form using the completed smoke models.
  await ev('location.hash="/reports/new"');
  await waitFor('document.querySelectorAll(".model-choice").length===2');
  for (const text of ["Calibration", "Agreement between models", "Weights"])
    await ev(
      `(()=>{const label=[...document.querySelectorAll('.analysis-choice>label')].find(e=>e.textContent.trim()===${JSON.stringify(text)});if(label?.querySelector('input').checked)label.querySelector('input').click();})()`,
    );
  await click("Build report →");
  await waitFor(
    'location.hash.startsWith("#/reports/")&&location.hash!=="#/reports/new"',
  );
  results.reportId = await ev("location.hash.slice(10)");
  await waitFor(
    'document.querySelector(".page-header .badge")?.textContent.trim()==="completed"',
    45000,
  );
  await waitFor('!!document.querySelector(".report-markdown table")');
  await shot("20-created-report");
  results.errors = events.filter((e) => e.method === "Runtime.exceptionThrown");
  await writeFile("ui/out/actions.json", JSON.stringify(results, null, 2));
  console.log(
    JSON.stringify(
      {
        runs: results.runs,
        pauseResume: results.pauseResume,
        reportId: results.reportId,
        states: results.finalRuns.map((r) => r.status.state),
        errors: results.errors,
      },
      null,
      2,
    ),
  );
} finally {
  close();
}
