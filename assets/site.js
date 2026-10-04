const taskMeta = [
  {
    id: 1,
    title: "Data audit",
    body: "Retained-trip counts, quality flags, and sharing-status denominators for all downstream tasks.",
    summary: "04_Results/Task_1/summary.json",
    table: "04_Results/Task_1/tables/task1_2_counts.csv",
    figure: "04_Results/report_figures/main/task1_service_status_overview.png"
  },
  {
    id: 2,
    title: "Demand and OD flows",
    body: "Pickup/dropoff hotspots and directed OD rankings by trip volume and passenger spending.",
    summary: "04_Results/Task_2/summary.json",
    table: "04_Results/Task_2/tables/top10_volume_od.csv",
    figure: "04_Results/report_figures/main/task2_demand_april.png"
  },
  {
    id: 3,
    title: "Pooling requests and matches",
    body: "Platform-level request rates, reported match rates, matching success, and zone thresholds.",
    summary: "04_Results/Task_3/summary.json",
    table: "04_Results/Task_3/tables/monthly_rates.csv",
    figure: "04_Results/report_figures/main/task3_ride_pooling_metrics.png"
  },
  {
    id: 4,
    title: "Travel-time comparison",
    body: "Uber Y/Y trips compared with supported N/N reference cells by OD, hour, and day type.",
    summary: "04_Results/Task_4/summary.json",
    table: "04_Results/Task_4/tables/month_summary.csv",
    figure: "04_Results/report_figures/main/task4_travel_time_difference.png"
  },
  {
    id: 5,
    title: "Passenger-spending comparison",
    body: "Observed Y/Y fares compared with N/N reference means, with day-clustered uncertainty.",
    summary: "04_Results/Task_5/summary.json",
    table: "04_Results/Task_5/tables/price_comparison.csv",
    figure: "04_Results/report_figures/main/task5_passenger_spending_difference.png"
  }
];

const figures = [
  ["1", "Dataset overview", "04_Results/report_figures/main/task1_service_status_overview.png", "Retained service counts and HVFHV request/match states."],
  ["2", "April demand", "04_Results/report_figures/main/task2_demand_april.png", "Pickup and dropoff concentration across Yellow Taxi and HVFHV trips."],
  ["2", "May demand", "04_Results/report_figures/main/task2_demand_may.png", "The same spatial demand comparison for May 2026."],
  ["2", "OD ranking comparison", "04_Results/report_figures/main/task2_od_flow_comparison.png", "Directed OD relationships by volume and aggregate spending."],
  ["3", "Pooling metrics", "04_Results/report_figures/main/task3_ride_pooling_metrics.png", "Request, reported match, and matching success rates for Uber and Lyft."],
  ["4", "Travel-time difference", "04_Results/report_figures/main/task4_travel_time_difference.png", "Supported shared trips compared with non-shared reference cells."],
  ["4", "Hourly coverage", "04_Results/report_figures/main/task4_hourly_pattern_coverage.png", "Hourly excess-time and reference-coverage patterns."],
  ["5", "Fare difference", "04_Results/report_figures/main/task5_passenger_spending_difference.png", "Passenger-spending differences with confidence intervals."]
];

const formatCompact = new Intl.NumberFormat("en", {
  notation: "compact",
  maximumFractionDigits: 1
});

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function renderTasks() {
  const grid = document.querySelector("#taskGrid");
  grid.innerHTML = taskMeta.map((task) => `
    <article class="task-card">
      <span class="task-number">${task.id}</span>
      <h3>${task.title}</h3>
      <p>${task.body}</p>
      <div class="task-links">
        <a href="${task.summary}">Summary</a>
        <a href="${task.table}">CSV</a>
        <a href="${task.figure}">Figure</a>
      </div>
    </article>
  `).join("");
}

function renderFigures() {
  const grid = document.querySelector("#figureGrid");
  grid.innerHTML = figures.map(([task, title, src, text]) => `
    <article class="figure-card" data-task="${task}">
      <a href="${src}"><img loading="lazy" src="${src}" alt="${title}"></a>
      <div class="figure-caption">
        <span class="tag">Task ${task}</span>
        <h3>${title}</h3>
        <p>${text}</p>
      </div>
    </article>
  `).join("");
}

function bindFigureFilters() {
  const buttons = document.querySelectorAll("[data-filter]");
  const cards = document.querySelectorAll(".figure-card");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const filter = button.dataset.filter;
      buttons.forEach((item) => item.classList.toggle("active", item === button));
      cards.forEach((card) => {
        card.hidden = filter !== "all" && card.dataset.task !== filter;
      });
    });
  });
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Unable to load ${path}`);
  }
  return response.json();
}

async function hydrateMetrics() {
  try {
    const [task1, task3, task4, task5] = await Promise.all([
      loadJson("04_Results/Task_1/summary.json"),
      loadJson("04_Results/Task_3/summary.json"),
      loadJson("04_Results/Task_4/summary.json"),
      loadJson("04_Results/Task_5/summary.json")
    ]);

    const retained = task1.files.reduce(
      (sum, item) => sum + Number(item.completed_trip_count_retained || 0),
      0
    );
    document.querySelector('[data-kpi="rawRecords"]').textContent = formatCompact.format(task1.raw_records);
    document.querySelector('[data-kpi="retainedTrips"]').textContent = formatCompact.format(retained);

    const uberApril = task3.monthly.find((row) => row.month === "2026-04" && row.service === "Uber");
    const uberMay = task3.monthly.find((row) => row.month === "2026-05" && row.service === "Uber");
    document.querySelector('[data-insight="task3"]').textContent =
      `Uber pooling requests were ${formatPercent(uberApril.pooling_request_rate)} in April and ${formatPercent(uberMay.pooling_request_rate)} in May, with reported matching success above 56% in both months.`;

    const medians = task4.month_summary.map((row) => row.excess_median_minutes.toFixed(1));
    document.querySelector('[data-insight="task4"]').textContent =
      `Supported Uber shared trips had median excess travel time of ${medians[0]} minutes in April and ${medians[1]} minutes in May.`;

    const uberFareDiffs = task5.comparison
      .filter((row) => row.service === "Uber")
      .map((row) => row.mean_difference_usd.toFixed(2));
    document.querySelector('[data-insight="task5"]').textContent =
      `Uber supported shared fares were ${uberFareDiffs[0]} USD and ${uberFareDiffs[1]} USD below comparable non-shared references in April and May.`;
  } catch (error) {
    console.warn(error);
  }
}

renderTasks();
renderFigures();
bindFigureFilters();
hydrateMetrics();
