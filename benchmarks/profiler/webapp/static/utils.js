// Storage for selected points (multi-selection)
const selectedPointKeys = {
    prefill: [],
    decode: [],
    cost: []
};

// Storage for all data points
const allDataPoints = {
    prefill: [],
    decode: [],
    cost: []
};

// Lookup from point key to row values
const pointDataLookup = {
    prefill: {},
    decode: {},
    cost: {}
};

const tableHeaders = {
    prefill: ["GPUs", "TTFT (ms)", "Throughput (tokens/s/GPU)"],
    decode: ["GPUs", "ITL (ms)", "Throughput (tokens/s/GPU)"],
    cost: [
        "TTFT (ms)",
        "Prefill Thpt (tokens/s/GPU)",
        "ITL (ms)",
        "Decode Thpt (tokens/s/GPU)",
        "Tokens/User",
        "Cost ($)"
    ]
};

function getTraceUid(trace, fallbackIndex) {
    if (!trace) {
        return `trace-${fallbackIndex}`;
    }
    return trace.uid || `trace-${fallbackIndex}`;
}

function makePointKey(traceUid, pointIndex) {
    return `${traceUid}:${pointIndex}`;
}

function getDisplayRows(plotType) {
    if (!selectedPointKeys[plotType] || selectedPointKeys[plotType].length === 0) {
        return allDataPoints[plotType].map((row) => row.values);
    }

    const lookup = pointDataLookup[plotType] || {};
    return selectedPointKeys[plotType]
        .map((key) => lookup[key])
        .filter(Boolean)
        .map((row) => row.values);
}

function computeSelectedKeys(plotDiv, lookup) {
    const keys = [];
    if (!plotDiv || !plotDiv.data) {
        return keys;
    }

    plotDiv.data.forEach((trace, traceIdx) => {
        if (!trace) {
            return;
        }

        const traceUid = getTraceUid(trace, traceIdx);
        const selectedPoints = trace.selectedpoints;

        if (!Array.isArray(selectedPoints) || selectedPoints.length === 0) {
            return;
        }

        selectedPoints.forEach((pointIndex) => {
            const key = makePointKey(traceUid, pointIndex);
            if (!lookup || lookup[key]) {
                keys.push(key);
            }
        });
    });

    return keys;
}

function normalizeRow(row) {
    if (row == null) {
        return [];
    }
    if (Array.isArray(row)) {
        return row.slice();
    }
    if (typeof row === "object") {
        if (typeof row[Symbol.iterator] === "function") {
            return Array.from(row);
        }
        return Object.values(row);
    }
    return [row];
}

function formatCell(value) {
    if (value == null) {
        return "";
    }
    if (typeof value === "number" && Number.isFinite(value)) {
        if (Number.isInteger(value)) {
            return value.toString();
        }
        return value.toFixed(3);
    }
    return `${value}`;
}

function renderTableHTML(headers, rows) {
    const safeHeaders = headers || [];
    const headerCells = safeHeaders.map((header) => `<th>${header}</th>`).join("");

    let bodyHtml = "";
    if (!rows || rows.length === 0) {
        bodyHtml = `<tr><td class="dynamo-table-empty" colspan="${safeHeaders.length || 1}">No data selected yet. Click points on the plot to populate this table.</td></tr>`;
    } else {
        bodyHtml = rows
            .map((row) => {
                const normalized = normalizeRow(row);
                const length = safeHeaders.length > 0 ? safeHeaders.length : normalized.length;
                const cells = Array.from({ length }, (_, idx) => {
                    const value = normalized[idx];
                    return `<td>${formatCell(value)}</td>`;
                });
                return `<tr>${cells.join("")}</tr>`;
            })
            .join("");
    }

    return `
        <div class="dynamo-table-wrapper">
            <table class="dynamo-table">
                <thead><tr>${headerCells}</tr></thead>
                <tbody>${bodyHtml}</tbody>
            </table>
        </div>
    `;
}

function updateDataTable(tableId, data, plotType) {
    const container = document.getElementById(tableId);
    if (!container) {
        console.log(`Table container ${tableId} not found`);
        return;
    }

    const headers = tableHeaders[plotType] || [];
    container.innerHTML = renderTableHTML(headers, data);
    console.log(`Updated table ${tableId} with ${data ? data.length : 0} rows`);
}

function resizePlotlyGraphs() {
    const plots = document.querySelectorAll('.js-plotly-plot');
    console.log(`Found ${plots.length} Plotly graphs`);
    for (let i = 0; i < plots.length; i++) {
        if (window.Plotly && plots[i]) {
            window.Plotly.relayout(plots[i], {autosize: true});
            console.log(`Resized plot ${i}`);
        }
    }
}

function setupPlotClickHandler(plotId, tableId, plotType) {
    const attemptSetup = () => {
        const plotContainer = document.querySelector(`#${plotId}`);
        if (!plotContainer) {
            console.log(`Plot ${plotId} not found, retrying...`);
            setTimeout(attemptSetup, 500);
            return;
        }

        const plotDiv = plotContainer.querySelector('.js-plotly-plot');
        if (!plotDiv) {
            console.log(`Plotly div not found in ${plotId}, retrying...`);
            setTimeout(attemptSetup, 500);
            return;
        }

        console.log(`Setting up handlers for ${plotId}`);

        const headers = tableHeaders[plotType] || [];

        const syncSelection = (source) => {
            const lookup = pointDataLookup[plotType] || {};
            const keys = computeSelectedKeys(plotDiv, lookup);
            selectedPointKeys[plotType] = keys;
            updateDataTable(tableId, getDisplayRows(plotType), plotType);
            console.log(`Selection synced for ${plotType} (${source || 'update'}): ${keys.length} point(s)`);
        };

        const refreshAllDataPoints = () => {
            if (!plotDiv || !plotDiv.data) {
                return;
            }

            const rows = [];
            const lookup = {};
            plotDiv.data.forEach((trace, traceIdx) => {
                if (!trace || !trace.customdata) {
                    return;
                }

                const traceUid = getTraceUid(trace, traceIdx);

                trace.customdata.forEach((item, pointIndex) => {
                    const normalized = normalizeRow(item);
                    if (normalized.length === 0) {
                        return;
                    }

                    const alignedRow = headers.length
                        ? headers.map((_, idx) => normalized[idx])
                        : normalized;

                    const key = makePointKey(traceUid, pointIndex);
                    const rowObj = { key, values: alignedRow };
                    rows.push(rowObj);
                    lookup[key] = rowObj;
                });
            });

            const newHash = JSON.stringify(rows.map((row) => [row.key, row.values]));
            if (plotDiv.__dynamo_data_hash !== newHash) {
                plotDiv.__dynamo_data_hash = newHash;
                allDataPoints[plotType] = rows;
                pointDataLookup[plotType] = lookup;
                syncSelection('data-refresh');
                console.log(`Stored ${rows.length} data points for ${plotType}`);
            }
        };

        refreshAllDataPoints();

        if (plotDiv.on) {
            plotDiv.on('plotly_afterplot', refreshAllDataPoints);
            plotDiv.on('plotly_restyle', refreshAllDataPoints);
            plotDiv.on('plotly_relayout', refreshAllDataPoints);
        }

        plotDiv.on('plotly_click', function(data) {
            console.log(`Click detected on ${plotId}`, data);
            if (data.points && data.points.length > 0) {
                setTimeout(() => syncSelection('click'), 0);
            }
        });

        if (plotDiv.on) {
            plotDiv.on('plotly_selected', function(eventData) {
                if (!eventData || !eventData.points) {
                    return;
                }

                syncSelection('selection-tool');
            });

            plotDiv.on('plotly_deselect', function() {
                syncSelection('deselect');
            });
        }

        console.log(`Handlers configured for ${plotId}`);
    };

    setTimeout(attemptSetup, 500);
}

// Wait for DOM to be ready and set up observers
setTimeout(() => {
    // Find all tab buttons and add click listeners
    const tabButtons = document.querySelectorAll('button[role="tab"]');
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            setTimeout(resizePlotlyGraphs, 150);
        });
    });

    // Use MutationObserver to detect tab visibility changes
    const observer = new MutationObserver(() => {
        resizePlotlyGraphs();
    });

    // Observe changes to elements with tab content
    const tabPanels = document.querySelectorAll('[role="tabpanel"]');
    tabPanels.forEach(panel => {
        observer.observe(panel, {
            attributes: true,
            attributeFilter: ['style', 'class', 'hidden']
        });
    });

    // Initial resize
    resizePlotlyGraphs();

    // Setup click handlers for all plots
    setupPlotClickHandler('prefill_plot', 'prefill_table', 'prefill');
    setupPlotClickHandler('decode_plot', 'decode_table', 'decode');
    setupPlotClickHandler('cost_plot', 'cost_table', 'cost');
}, 1000);

// Also resize on window resize
window.addEventListener('resize', resizePlotlyGraphs);