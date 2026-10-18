// Screenshots of the running app, one per step, and the page errors met on the way.
//   node scripts/shots.mjs OUT_DIR name=/path name=js:EXPRESSION ...
// A path step navigates; a js: step evaluates in the page (e.g. opens a dialog). Each
// step waits for the page to settle, then captures the window.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { connect } from "./cdp.mjs";

const [out = "out", ...steps] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
const page = await connect();
const errors = [];
await page.send("Runtime.enable");
page.on("Runtime.exceptionThrown", ({ exceptionDetails }) =>
  errors.push(exceptionDetails.exception?.description ?? exceptionDetails.text),
);
page.on("Runtime.consoleAPICalled", ({ type, args }) => {
  if (type === "error") errors.push(args.map((arg) => arg.value ?? arg.description).join(" "));
});
const settle = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
for (const step of steps) {
  const [name, ...rest] = step.split("=");
  const action = rest.join("=");
  if (action.startsWith("js:")) await page.evaluate(action.slice(3));
  else await page.evaluate(`navigation.navigate(${JSON.stringify(action)}).finished.then(() => true)`);
  await settle(900);
  const { data } = await page.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(out, `${name}.png`), Buffer.from(data, "base64"));
  console.log(`${name}: ${join(out, `${name}.png`)}`);
}
console.log(JSON.stringify({ errors }, null, 1));
page.close();
