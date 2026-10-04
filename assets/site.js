const tasks = [
  {id:1, short:"Data audit", title:"Establishing the denominator", body:"The four monthly files are audited and screened before any service, pooling, or comparison result is calculated. Retained HVFHV sharing states reconcile to the monthly totals.", stat:"50.9M trips", note:"Completed trips retained", figure:"04_Results/report_figures/main/task1_service_status_overview.png", caption:"Retained service counts and sharing status overview."},
  {id:2, short:"Demand & OD", title:"Where trips begin and end", body:"Yellow Taxi demand is concentrated in Manhattan. HVFHV demand extends more broadly across the city, with strong airport-linked movements and distinct directed OD patterns.", stat:"8 maps", note:"Pickup and dropoff views", figure:"04_Results/report_figures/main/task2_demand_april.png", caption:"April pickup and dropoff density across both services."},
  {id:3, short:"Pooling", title:"Requests are not matches", body:"Reported shared requests and completed matches are kept separate. Uber dominates pooling activity while Lyft's shared samples are much smaller.", stat:"2.5-2.7%", note:"Uber shared-request rate", figure:"04_Results/report_figures/main/task3_ride_pooling_metrics.png", caption:"Request, reported match, and matching success rates."},
  {id:4, short:"Travel time", title:"The time cost of a shared ride", body:"Supported Uber Y/Y trips are compared with non-shared trips in matching OD-time reference cells. The median excess is about five minutes in each month.", stat:"+4.5 to +5.0 min", note:"Median excess travel time", figure:"04_Results/report_figures/main/task4_travel_time_difference.png", caption:"Travel-time difference for supported comparisons."},
  {id:5, short:"Fare & potential", title:"Lower fares and nearby trips", body:"Supported shared rides have lower passenger spending than comparable non-shared rides. A separate screen finds same-OD temporal neighbors among many N/N trips; this does not guarantee a feasible match.", stat:"45% Uber N/N", note:"10-minute co-occurrence screen", figure:"04_Results/Task_5/figures/potential_sharing.png", caption:"Potential sharing under the same-OD time-window rule."}
];

const figures = [
  [1,"Dataset overview","04_Results/report_figures/main/task1_service_status_overview.png","Retained service counts and HVFHV request/match states."],
  [2,"April demand","04_Results/report_figures/main/task2_demand_april.png","Pickup and dropoff concentration across Yellow Taxi and HVFHV."],
  [2,"May demand","04_Results/report_figures/main/task2_demand_may.png","The same spatial demand comparison for May 2026."],
  [2,"OD ranking comparison","04_Results/report_figures/main/task2_od_flow_comparison.png","Directed OD relationships by volume and aggregate spending."],
  [3,"Pooling metrics","04_Results/report_figures/main/task3_ride_pooling_metrics.png","Request, reported match, and matching success rates."],
  [4,"Travel-time difference","04_Results/report_figures/main/task4_travel_time_difference.png","Supported shared trips compared with non-shared reference cells."],
  [4,"Hourly coverage","04_Results/report_figures/main/task4_hourly_pattern_coverage.png","Hourly excess-time and reference-coverage patterns."],
  [5,"Fare difference","04_Results/report_figures/main/task5_passenger_spending_difference.png","Passenger-spending differences with confidence intervals."],
  [5,"Potential sharing","04_Results/Task_5/figures/potential_sharing.png","N/N trips with a same-platform, same-OD temporal neighbor."]
];

const compact = new Intl.NumberFormat("en", {notation:"compact", maximumFractionDigits:1});
const taskTabs = document.querySelector("#taskTabs");
const taskDetail = document.querySelector("#taskDetail");
const figureGrid = document.querySelector("#figureGrid");
const dialog = document.querySelector("#figureDialog");
let currentTask = 1;
let returnFocus = null;

function formatPercent(value) { return `${(value * 100).toFixed(1)}%`; }

function openFigure(src, title, description, task, trigger) {
  returnFocus = trigger;
  document.querySelector("#dialogTask").textContent = `Task ${task} / Figure`;
  document.querySelector("#dialogTitle").textContent = title;
  document.querySelector("#dialogImage").src = src;
  document.querySelector("#dialogImage").alt = title;
  document.querySelector("#dialogDescription").textContent = description;
  document.querySelector("#dialogOriginal").href = src;
  dialog.showModal();
  document.querySelector("#closeDialog").focus();
}

function selectTask(id, focusTab = false) {
  const task = tasks.find((item) => item.id === id);
  if (!task) return;
  currentTask = id;
  taskTabs.querySelectorAll(".task-tab").forEach((tab) => {
    const selected = Number(tab.dataset.task) === id;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    if (selected && focusTab) tab.focus();
  });
  taskDetail.setAttribute("aria-labelledby", `task-tab-${id}`);
  taskDetail.innerHTML = `<div class="task-copy"><span class="task-kicker">Task ${id} / ${task.short}</span><h3>${task.title}</h3><p>${task.body}</p><div class="task-evidence"><strong>${task.stat}</strong><span>${task.note}</span></div><a href="main.pdf">Evidence in the full report <span aria-hidden="true">&#8599;</span></a></div><div class="task-media"><button type="button" aria-label="Expand Task ${id} figure"><img src="${task.figure}" alt="${task.caption}"></button></div>`;
  // Replacing the panel makes each task change a distinct, short transition.
  taskDetail.style.animation = "none";
  taskDetail.offsetHeight;
  taskDetail.style.animation = "";
  taskDetail.querySelector(".task-media button").addEventListener("click", (event) => openFigure(task.figure, task.title, task.caption, id, event.currentTarget));
}

function renderTasks() {
  taskTabs.innerHTML = tasks.map((task) => `<button type="button" class="task-tab" id="task-tab-${task.id}" role="tab" aria-controls="taskDetail" aria-selected="false" tabindex="-1" data-task="${task.id}"><span>0${task.id}</span>${task.short}</button>`).join("");
  taskTabs.addEventListener("click", (event) => {
    const tab = event.target.closest(".task-tab");
    if (tab) selectTask(Number(tab.dataset.task));
  });
  taskTabs.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? 1 : event.key === "End" ? tasks.length : Math.min(tasks.length, Math.max(1, currentTask + (event.key === "ArrowRight" ? 1 : -1)));
    selectTask(next, true);
  });
  document.querySelectorAll("[data-go-task]").forEach((link) => link.addEventListener("click", () => selectTask(Number(link.dataset.goTask))));
  selectTask(1);
}

function renderFigures() {
  figureGrid.innerHTML = figures.map(([task,title,src,description]) => `<article class="figure-card" data-task="${task}"><button type="button" aria-label="Expand ${title}"><img loading="lazy" src="${src}" alt="${title}"></button><div class="figure-caption"><span>Task ${task}</span><h3>${title}</h3><p>${description}</p></div></article>`).join("");
  figureGrid.querySelectorAll(".figure-card").forEach((card, index) => {
    card.querySelector("button").addEventListener("click", (event) => {
      const [task,title,src,description] = figures[index];
      openFigure(src,title,description,task,event.currentTarget);
    });
  });
  document.querySelector("#figureFilters").addEventListener("click", (event) => {
    const button = event.target.closest("[data-filter]");
    if (!button) return;
    const filter = button.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach((item) => {
      const active = item === button;
      item.classList.toggle("active",active);
      item.setAttribute("aria-pressed",String(active));
    });
    const visible = figures.filter(([task]) => filter === "all" || String(task) === filter).length;
    figureGrid.querySelectorAll(".figure-card").forEach((card) => { card.hidden = filter !== "all" && card.dataset.task !== filter; });
    document.querySelector("#figureCount").textContent = `${visible} ${visible === 1 ? "figure" : "figures"}`;
  });
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Unable to load ${path}`);
  return response.json();
}

async function hydrateMetrics() {
  try {
    const [task1,task3,task4,task5] = await Promise.all([
      loadJson("04_Results/Task_1/summary.json"), loadJson("04_Results/Task_3/summary.json"),
      loadJson("04_Results/Task_4/summary.json"), loadJson("04_Results/Task_5/summary.json")
    ]);
    const retained = task1.files.reduce((sum,item) => sum + Number(item.completed_trip_count_retained || 0), 0);
    document.querySelector('[data-kpi="rawRecords"]').textContent = compact.format(task1.raw_records);
    document.querySelector('[data-kpi="retainedTrips"]').textContent = compact.format(retained);
    const uberApril = task3.monthly.find((row) => row.month === "2026-04" && row.service === "Uber");
    const uberMay = task3.monthly.find((row) => row.month === "2026-05" && row.service === "Uber");
    document.querySelector('[data-insight="task3"]').textContent = `Uber pooling requests were ${formatPercent(uberApril.pooling_request_rate)} in April and ${formatPercent(uberMay.pooling_request_rate)} in May, with reported matching success above 56% in both months.`;
    const medians = task4.month_summary.map((row) => row.excess_median_minutes.toFixed(1));
    document.querySelector('[data-insight="task4"]').textContent = `Supported Uber shared trips had median excess travel time of ${medians[0]} minutes in April and ${medians[1]} minutes in May.`;
    const fareDiffs = task5.comparison.filter((row) => row.service === "Uber").map((row) => Math.abs(row.mean_difference_usd).toFixed(2));
    const potential = task5.task5_2.main_results.filter((row) => row.service === "Uber").map((row) => formatPercent(row.potentially_shareable_share));
    document.querySelector('[data-insight="task5"]').textContent = `Supported Uber shared fares were ${fareDiffs[0]} USD and ${fareDiffs[1]} USD below comparable non-shared references; ${potential[0]} and ${potential[1]} of Uber N/N trips had same-OD 10-minute neighbors.`;
  } catch (error) { console.warn(error); }
}

document.querySelector("#closeDialog").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
dialog.addEventListener("close", () => { document.querySelector("#dialogImage").removeAttribute("src"); returnFocus?.focus(); });
renderTasks();
renderFigures();
hydrateMetrics();
