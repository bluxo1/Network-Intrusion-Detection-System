# Manual: from the NSL-KDD demo to a live IDS pilot

This is a step-by-step plan for a **new live-traffic pilot**. The current Flask
app and its saved models accept 41 NSL-KDD features; they do not collect packets
or accept Zeek records. Keep the current demo as a benchmark while building a
separate live-data pipeline. The NSL-KDD maintainers caution that the dataset
may not represent real networks [1].

**Windows-only, own-PC pilot:** use the [Windows procedure](windows_pc_pilot.md)
for PktMon collection and Zeek analysis in Ubuntu WSL, then return to steps
7–10 below. Steps 2–4 here describe a separate Linux sensor path.

**Who does what:** The operator (you) chooses the monitored network, collects
authorized data, supplies labels, and runs the pilot. The developer (Codex)
implements the new feature pipeline, trains and evaluates a model, and connects
it to the app after the data handoff. Both agree on acceptance targets before
looking at final test results.

## 1. Define the pilot scope (operator)

Write down these five items before capturing traffic:

1. The network or lab segment you are permitted to monitor, and who owns it.
2. The sensor location and which traffic it can actually see. A sensor on one
   workstation does not automatically see the rest of the network.
3. The attack behaviors that matter, especially what **R2L** means in this
   environment (for example, unauthorized account access).
4. The operator who will review alerts and the maximum false alerts per day
   that person can handle.
5. The minimum acceptable detection rate for each priority behavior, plus the
   maximum alert delay. Record these targets before model experiments.

**Output:** a short `pilot_scope.md` outside this Git repository. Start with a
controlled lab or an approved pilot segment. Do not call a model production
ready solely because it performs well on NSL-KDD.

## 2. Choose one input format (operator, with developer review)

For the first pilot, use **Zeek JSON logs** from a Linux sensor. Zeek can read
stored PCAP files or a live interface, and its `conn.log` gives one record per
observed connection [2, 3]. This is a concrete starting point, not a way to
reuse the current 41-feature model. The developer will train a new model for
the actual log schema.

For R2L, include authentication or host events when available. Zeek connection
metadata describes who connected to whom, but cannot by itself establish
whether a login succeeded. On Windows hosts, Security events **4624** and
**4625** record successful and failed logons respectively [4, 5]. The feature
pipeline can correlate those events with connections where the evidence permits.

**Output:** a list of chosen log sources and sample field names. If you choose
a different collector, stop here and agree on its schema before proceeding.

## 3. Check the collector with an offline capture (operator)

Install Zeek on a Linux machine using the official installation guide [6].
Use a PCAP from traffic you are authorized to analyze. Work in a new directory
so generated logs do not overwrite earlier results:

```bash
zeek --version
mkdir -p ~/nids-pilot/zeek/offline-01
cd ~/nids-pilot/zeek/offline-01
zeek -r "/path/to/authorized-capture.pcap" LogAscii::use_json=T
test -s conn.log
head -n 1 conn.log
```

Each line of `conn.log` should be one JSON object with fields such as `ts`,
`uid`, `proto`, `service`, `duration`, and originator/responder addresses [2].
Other logs such as `dns.log` or `http.log` appear only when the capture contains
traffic Zeek can analyze for those protocols. Retain the Zeek version and the
command used, since feature meanings can change across versions.

**Output:** one nonempty `conn.log`, the Zeek version, and a field inventory.

## 4. Capture a small live pilot (operator)

After the offline check, connect the Linux sensor to the approved traffic
source. Use the interface that actually receives the monitored traffic. Run
Zeek in a separate clean directory for a short observation window:

```bash
mkdir -p ~/nids-pilot/zeek/live-01
cd ~/nids-pilot/zeek/live-01
zeek -i enp3s0 -C LogAscii::use_json=T
```

Replace `enp3s0` with the approved capture interface and stop the command with
Ctrl+C after the observation window. Zeek's quick start documents interface
capture and uses `-C` to handle local checksum offloading [3]. Verify that
`conn.log` grows during known activity, check the
sensor clock, and note packet loss or missing traffic if observed. Keep packet
captures and full logs outside this Git repository.

**Output:** timestamped logs and a capture note containing sensor, interface,
time range, Zeek version, and known gaps. Extend collection to cover normal
operating patterns before judging false-alert rates.

## 5. Create ground-truth labels (operator)

Create a CSV next to the logs, with this header:

```csv
capture_id,uid,ts,label,attack_family,attack_subtype,evidence_ref
```

- `capture_id` identifies the capture session; `uid` and `ts` identify a Zeek
  connection within it.
- `label` is `normal`, `attack`, or `unknown`. Do not treat an unreviewed record
  as normal merely because no existing tool raised an alert.
- Fill `attack_family` and `attack_subtype` only when supported by evidence.
  Keep the evidence reference in your own access-controlled notes.
- Use controlled lab scenarios or analyst-reviewed incidents for attack
  labels. Model predictions are not ground truth.
- If host events cannot be matched to an individual connection, label a
  documented time window or incident separately; do not invent a `uid` match.

For a Windows host, this PowerShell check confirms whether recent logon events
are available without exporting their full messages:

```powershell
Get-WinEvent -FilterHashtable @{LogName='Security'; Id=4624,4625; StartTime=(Get-Date).AddDays(-1)} -MaxEvents 20 |
  Select-Object TimeCreated, Id, MachineName
```

Audit settings and permissions determine whether these events are present.
Record timestamps in UTC, retain the time zone used by each source, and keep
raw event messages and credentials out of Git.

**Output:** a label file with documented provenance, plus counts for normal,
attack, unknown, and each supported attack subtype.

## 6. Prepare the handoff (operator to developer)

Share a small **redacted** sample first, not the whole capture. The handoff
should contain:

```text
pilot_scope.md
capture_notes.md
schema.md                 # fields, types, units, missing-value meaning
sample_conn.jsonl         # a few representative, redacted records
sample_auth_events.jsonl  # only if host events are in scope
labels.csv                # labels for sample records, where known
counts.md                 # row counts by time period and label
```

Include the acceptable false alerts per day and priority detection targets
from step 1. Remove credentials, payloads, and unnecessary personal identifiers
before sharing. Keep full PCAPs and raw security logs in a controlled location,
not in the public repository.

**Handoff gate:** the developer can parse the sample, identify missing fields,
and join the supplied labels to records without guessing.

## 7. Build and validate the new model (developer)

After the handoff, the developer will:

1. Define a versioned schema for the collected fields. Reject malformed or
   unsupported records explicitly instead of filling unknown fields with
   values borrowed from NSL-KDD.
2. Create a deterministic parser and feature extractor. Fit encoders and
   scalers on training data only, then reuse them unchanged for validation and
   serving.
3. Separate **training**, **threshold selection**, and **final test** data.
   Use later time periods and, where possible, whole attack scenarios or sites
   as holdouts. Group/time-based evaluation is important when neighboring
   records are correlated [7].
4. Compare a simple baseline with candidate models. Report attack recall by
   subtype, false alerts per day, precision, missing-field rate, and latency.
   Do not select a model or threshold on the final test set.
5. Document the model version, training-data period, schema version, chosen
   threshold, and limitations. Promote a candidate only if it meets the targets
   agreed in step 1 on held-out data.

**Output:** a reproducible training command, model artifacts for the new
schema, and a report showing both successes and failure cases. The present
`src/train.py`, `src/preprocess.py`, and `/api/predict` are NSL-KDD-specific;
this step requires new code rather than feeding Zeek logs into them unchanged.

## 8. Connect ingestion and run in observation mode (developer + operator)

Build this path with explicit schema and model versions:

```text
Zeek/host logs -> parser -> feature record -> new model -> timestamped alert
```

Each alert should carry a record ID, event time, model version, score, and
reason for any rejected input. Keep the pilot **observation only**: an alert
goes to a reviewer and does not automatically block traffic. Replay a known
capture through the whole path, then run a live pilot and verify that record
counts, IDs, and timestamps match from collector to alert.

**Output:** a repeatable end-to-end replay and a live observation period with
measured ingestion failures, alert delay, and throughput.

## 9. Review alerts and monitor performance (operator + developer)

Review a sample of alerts and non-alerts and record outcomes. Track at least:

- Captured records, parsed records, dropped/invalid records, and missing fields.
- Alerts per day, reviewed false alerts per day, and detection by priority
  subtype when verified labels are available.
- Prediction latency, data/score distribution changes, collector uptime, and
  the time between event and alert.

NIST recommends monitoring deployed AI performance and changes in the data it
sees [8]. When performance falls outside the agreed limits, investigate data
quality first, retrain on reviewed labels, and repeat the held-out evaluation
before replacing the model.

**Output:** a pilot report covering at least one full normal operating cycle,
with reviewed alerts and documented failure modes.

## 10. Decide whether the pilot is complete (joint decision)

The live IDS work is complete for the agreed pilot scope only when:

- The collector and feature schema are documented and reproducible.
- Held-out and live-pilot results meet the pre-agreed detection, false-alert,
  latency, and data-quality targets.
- Alerts can be traced to source records and reviewed by an operator.
- Model/version rollback and failure behavior have been exercised.
- The README accurately states the supported environment and remaining gaps.

If any gate fails, keep the system in observation mode and work on the failing
data or model behavior. A perfect score is not required; measured performance
against a stated use case is.

## Sources

1. [UNB: NSL-KDD dataset and limitations](https://www.unb.ca/cic/datasets/nsl.html)
2. [Zeek: `conn.log` fields and JSON examples](https://docs.zeek.org/en/v8.1.0/logs/conn.html)
3. [Zeek: quick start, PCAP and live interface](https://docs.zeek.org/en/master/quickstart.html)
4. [Microsoft: successful logon event 4624](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/auditing/event-4624)
5. [Microsoft: failed logon event 4625](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/auditing/event-4625)
6. [Zeek: installation](https://docs.zeek.org/en/current/install.html)
7. [Scikit-learn: group and time-based validation](https://scikit-learn.org/stable/modules/cross_validation.html)
8. [NIST AI RMF: measure and monitor](https://airc.nist.gov/airmf-resources/playbook/measure/)
