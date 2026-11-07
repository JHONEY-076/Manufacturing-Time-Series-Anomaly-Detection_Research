// Browser integration checks using the installed Chrome and its local DevTools protocol.
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";
import {spawn} from "node:child_process";
import assert from "node:assert/strict";

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),"..");
const output=path.join(root,"data/research11/ui_validation");
fs.mkdirSync(output,{recursive:true});
const profile=fs.mkdtempSync(path.join(output,"browser_profile_"));
const executable="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const browser=spawn(executable,["--headless=new","--remote-debugging-port=0",`--user-data-dir=${profile}`,"--no-first-run","--no-default-browser-check","--disable-background-networking","about:blank"],{windowsHide:true,stdio:["ignore","ignore","pipe"]});
let browserError="";browser.stderr.on("data",chunk=>{browserError=(browserError+chunk).slice(-5000);});
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
let socket;
const results={started_utc:new Date().toISOString(),checks:[],browser_errors:[],api_posts:[]};
async function until(callback,timeout=15000){const end=Date.now()+timeout;while(Date.now()<end){if(await callback())return;await sleep(100);}throw new Error("Browser check timeout");}
try{
  const portFile=path.join(profile,"DevToolsActivePort");
  await until(()=>fs.existsSync(portFile));
  const port=Number(fs.readFileSync(portFile,"utf8").split("\n")[0]);
  const target=await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`,{method:"PUT"})).json();
  socket=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
  let id=0;const pending=new Map();
  socket.onmessage=event=>{
    const message=JSON.parse(event.data);
    if(message.id){const pair=pending.get(message.id);if(!pair)return;pending.delete(message.id);clearTimeout(pair.timer);message.error?pair.reject(new Error(JSON.stringify(message.error))):pair.resolve(message.result);}
    if(message.method==="Runtime.exceptionThrown")results.browser_errors.push(message.params.exceptionDetails);
    if(message.method==="Network.requestWillBeSent" && message.params.request.url.endsWith("/compare"))results.api_posts.push({method:message.params.request.method,url:message.params.request.url});
  };
  socket.onclose=()=>{for(const pair of pending.values()){clearTimeout(pair.timer);pair.reject(new Error("Browser debugging connection closed: "+browserError));}pending.clear();};
  const send=(method,params={})=>new Promise((resolve,reject)=>{const key=++id;const timer=setTimeout(()=>{pending.delete(key);reject(new Error("DevTools timeout: "+method+" "+browserError));},15000);pending.set(key,{resolve,reject,timer});socket.send(JSON.stringify({id:key,method,params}));});
  const evaluate=async expression=>{const result=await send("Runtime.evaluate",{expression,awaitPromise:true,returnByValue:true});if(result.exceptionDetails)throw new Error(JSON.stringify(result.exceptionDetails));return result.result.value;};
  const check=(name)=>{results.checks.push(name);console.log("PASS",name);};
  const waitJS=expression=>until(()=>evaluate(expression));
  const screenshot=async filename=>{const result=await send("Page.captureScreenshot",{format:"png",captureBeyondViewport:true});fs.writeFileSync(path.join(output,filename),Buffer.from(result.data,"base64"));};
  await send("Page.enable");await send("Runtime.enable");await send("Network.enable");
  await send("Emulation.setDeviceMetricsOverride",{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await send("Page.navigate",{url:"http://127.0.0.1:8000/"});
  await waitJS(`document.getElementById('compare-button') && !document.getElementById('compare-button').disabled`);
  assert.match(await evaluate(`document.getElementById('connection').textContent`),/3개 모델/);check("dashboard loads all three models");
  await evaluate(`document.getElementById('compare-button').click()`);
  await waitJS(`document.getElementById('run-state').textContent.startsWith('완료')`);
  let response=await evaluate(`JSON.parse(document.getElementById('response-json').textContent)`);
  assert.deepEqual(Object.keys(response.predictions),["baseline","magnitude_warping","fmd"]);check("normal sample sends real POST /compare and renders three predictions");
  await evaluate(`document.getElementById('anomaly-sample').click()`);
  await waitJS(`!document.getElementById('compare-button').disabled && document.getElementById('truth').textContent.endsWith('이상')`);
  await evaluate(`document.getElementById('compare-button').click()`);
  await waitJS(`document.getElementById('run-state').textContent.startsWith('완료')`);
  response=await evaluate(`JSON.parse(document.getElementById('response-json').textContent)`);
  assert.equal(response.predictions.magnitude_warping.prediction,1);check("anomaly sample selection and prediction");
  await screenshot("dashboard_desktop.png");
  const posts=results.api_posts.length;
  await evaluate(`document.getElementById('request-json').value='{}';document.getElementById('request-json').dispatchEvent(new Event('input'));document.getElementById('compare-button').click()`);
  await waitJS(`!document.getElementById('global-error').hidden`);
  assert.equal(results.api_posts.length,posts);assert.match(await evaluate(`document.getElementById('truth').textContent`),/정답 정보 없음/);check("invalid custom input is rejected and old prediction cleared");
  const document=await send("DOM.getDocument");
  const fileInput=await send("DOM.querySelector",{nodeId:document.root.nodeId,selector:"#file-input"});
  await send("DOM.setFileInputFiles",{nodeId:fileInput.nodeId,files:[path.join(root,"data/research11/results/example_request.json")]});
  await waitJS(`document.getElementById('global-error').hidden && JSON.parse(document.getElementById('request-json').value).window.length===30`);
  await evaluate(`document.getElementById('compare-button').click()`);
  await waitJS(`document.getElementById('run-state').textContent.startsWith('완료')`);check("local JSON file can be loaded and submitted");
  const downloads=path.join(output,"downloads");fs.mkdirSync(downloads,{recursive:true});
  await send("Browser.setDownloadBehavior",{behavior:"allow",downloadPath:downloads});
  await evaluate(`document.getElementById('download-response').click()`);
  const saved=path.join(downloads,"weld-api-comparison.json");await until(()=>fs.existsSync(saved));
  assert.equal(JSON.parse(fs.readFileSync(saved,"utf8")).request.window.length,30);check("actual request and response JSON can be downloaded");
  await evaluate(`document.getElementById('normal-sample').click()`);
  await waitJS(`!document.getElementById('compare-button').disabled && document.getElementById('global-error').hidden`);
  const beforeReplay=results.api_posts.length;
  await evaluate(`document.getElementById('replay-button').click()`);
  await waitJS(`document.getElementById('run-state').textContent.startsWith('재생 완료')`);
  assert.equal(results.api_posts.length-beforeReplay,10);check("ten chronological windows replay through real HTTP");
  await evaluate(`document.getElementById('replay-button').click()`);
  await waitJS(`!document.getElementById('stop-button').hidden`);
  await evaluate(`document.getElementById('stop-button').click()`);
  await waitJS(`document.getElementById('run-state').textContent.startsWith('재생 중지')`);check("replay stop control");
  await evaluate(`document.getElementById('tab-results').click()`);
  await waitJS(`document.querySelector('.comparison-figure').naturalWidth>0`);
  assert.equal(await evaluate(`document.querySelectorAll('#metrics-table tr').length`),3);check("recorded benchmark table and comparison figure");
  await screenshot("dashboard_results.png");
  await evaluate(`document.getElementById('tab-live').click()`);
  await send("Emulation.setDeviceMetricsOverride",{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await sleep(200);
  assert.equal(await evaluate(`document.documentElement.scrollWidth <= window.innerWidth`),true);check("mobile dashboard has no page overflow");
  await screenshot("dashboard_mobile.png");
  await send("Page.navigate",{url:"http://127.0.0.1:8000/guide"});
  await waitJS(`document.body && document.body.textContent.includes('Start_API.cmd') && document.getElementById('stop')`);
  assert.match(await evaluate(`document.body.textContent`),/Stop_API.cmd/);check("detailed browser guide is available");
  assert.equal(results.browser_errors.length,0);check("no uncaught browser JavaScript exceptions");
  results.passed=true;
  await send("Browser.close").catch(()=>{});
}catch(error){results.passed=false;results.error=String(error);results.browser_stderr=browserError;process.exitCode=1;console.error(error);}
finally{
  socket?.close();
  if(browser.exitCode===null){await sleep(500);if(browser.exitCode===null)browser.kill();}
  results.finished_utc=new Date().toISOString();
  fs.writeFileSync(path.join(output,"browser_checks.json"),JSON.stringify(results,null,2));
  await sleep(500);
  // Remove only the exact temporary browser profile created by this run inside the workspace.
  const relative=path.relative(output,path.resolve(profile));
  if(relative.startsWith("browser_profile_") && !relative.includes(path.sep) && !path.isAbsolute(relative)){
    try{fs.rmSync(profile,{recursive:true,force:true});}catch{/* Chrome may briefly retain a file handle. */}
  }
}
