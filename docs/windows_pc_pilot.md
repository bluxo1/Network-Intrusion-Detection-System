# Windows-only pilot: this PC

Use this path when the monitored environment is **your own Windows PC** and no
Linux sensor is available. It narrows the first pilot to attempted unauthorized
logins against this PC. Windows Security events are host telemetry; optional
PktMon data describes only traffic seen by this PC. This is a pilot input for a
**new** model, not input to the repository's 41-feature NSL-KDD model.

The documented pilot used Zeek 8.0.10 in Ubuntu WSL at `/opt/zeek/bin/zeek`.
Its offline smoke test produced `conn.log` from Zeek's published sample capture. Use Zeek
to analyze a Windows capture **after** PktMon creates a PCAPNG file. WSL uses
a separate virtual network in its default NAT mode, so capturing on a WSL
interface is not proof of visibility into Windows host traffic [8].

With no second authorized device supplying remote attempts, this first pass
checks collection and labeling; it cannot measure true remote-to-local attack
recall yet. Do not enable a remote login service solely to create test traffic.

Record the current scope in
`C:\Users\<WindowsUser>\nids-pilot\pilot_scope.md` outside this repository.
The first review budget is five false alerts per day, with a five-minute alert
delay target. Treat those as pilot targets and measure actual results before
claiming detection quality.

## 1. Use a local working folder

Keep raw logs and captures outside this public Git repository. Use
`C:\Users\<WindowsUser>\nids-pilot` as the working folder and put the scope
draft there. Do not upload a raw ETL, PCAPNG, or full Security event export to
GitHub.

## 2. Run the read-only administrator preflight

An unelevated shell cannot query the Security log or talk to the PktMon driver.
Open **Windows PowerShell as Administrator** on
your PC and run the block below. It does not change audit settings or start a
capture:

```powershell
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
"elevated=$isAdmin"
auditpol /get /subcategory:"Logon"
Get-WinEvent -ListLog Security | Select-Object LogName, IsEnabled, RecordCount
pktmon status
```

Confirm `elevated=True`. Note whether Logon success and failure auditing are
enabled, whether the Security log is readable, and whether PktMon is already
running. Do **not** start another PktMon capture if one is active. If any
command fails, record its error text and stop here; the next action depends on
that result. PktMon and Windows Security event behavior are documented by
Microsoft [1, 2, 3].

**Send back:** the four command outputs above, with computer/account names
removed if you prefer. Do not send raw event messages or packet files.

## 3. Check whether logon events exist

After the preflight succeeds, run this in the same elevated PowerShell window:

```powershell
$events = @(Get-WinEvent -FilterHashtable @{
  LogName = 'Security'
  Id = 4624,4625
  StartTime = (Get-Date).AddDays(-1)
} -ErrorAction SilentlyContinue)
foreach ($id in 4624,4625) {
  [pscustomobject]@{ EventId = $id; Count = @($events | Where-Object Id -eq $id).Count }
}
```

Event **4624** means a logon session was created; **4625** means a failed
logon [4, 5]. The query reports only aggregate counts. No
matching events does not prove that no attempts occurred: check auditing,
retention, and the observation window. Do not change audit policy until its
current setting and the desired scope are reviewed.

**Output:** count of each event ID for the last 24 hours. The label
`normal` cannot be inferred merely from absence of a 4625 event.

## 4. Make one short packet capture, if network context is needed

Security events are the main source for the first login-focused pilot. A short
PktMon capture can show which connections this PC sees. Run this only after
step 2 succeeds, `pktmon status` shows no active capture, and the capture window
contains only traffic you intend to include. It creates a bounded, circular
64 MB ETL file and keeps the first 128 bytes of each packet [2]:

```powershell
$pilotRoot = Join-Path $env:USERPROFILE 'nids-pilot'
$captureDir = Join-Path $pilotRoot 'captures'
New-Item -ItemType Directory -Path $captureDir -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$etlPath = Join-Path $captureDir "pilot-$stamp.etl"
$pcapPath = Join-Path $captureDir "pilot-$stamp.pcapng"
pktmon start --capture --comp nics --pkt-size 128 --file-size 64 --file-name $etlPath
if ($LASTEXITCODE -ne 0) { throw 'PktMon did not start; no capture was made.' }
try { Start-Sleep -Seconds 60 } finally { pktmon stop }
pktmon etl2pcap $etlPath --out $pcapPath
Get-Item -LiteralPath $etlPath,$pcapPath | Select-Object Name, Length
```

PktMon's official commands support NIC-only capture, bounded ETL logs, and
PCAPNG conversion [2, 6]. This captures traffic from this PC's NICs, not an
entire home network. Keep both files local. If conversion fails, retain the
ETL and report the error; do not rerun a longer capture yet.

**Output:** file names, sizes, and exact capture start/end times. No packet
content needs to be sent at this stage.

### Analyze that capture with Zeek in Ubuntu WSL

From Windows PowerShell, enter Ubuntu with `wsl -d Ubuntu`. Then run these
commands in Ubuntu, replacing `<WindowsUser>` and the example PCAPNG file name
with your own values. Use a new output directory for each capture:

```bash
ls -lh /mnt/c/Users/<WindowsUser>/nids-pilot/captures/*.pcapng
mkdir -p ~/nids-pilot/zeek/pc-01
cd ~/nids-pilot/zeek/pc-01
/opt/zeek/bin/zeek -r "/mnt/c/Users/<WindowsUser>/nids-pilot/captures/pilot-YYYYMMDD-HHMMSS.pcapng" LogAscii::use_json=T
test -s conn.log
wc -l conn.log
```

Zeek's `-r` option reads captured traffic and the JSON option emits one
record per line [7]. If Zeek cannot read the converted file or `conn.log` is
empty, keep the ETL/PCAPNG files local and report the exact error. Do not feed
`conn.log` into the current `/api/predict` route; its feature schema is
different.

**Output:** Zeek command result, `conn.log` row count, and log file names.

## 5. Record labels without guessing

For each controlled test scenario, record its UTC start/end times, the test
account or service in private notes, what was attempted, and whether it was
an authorized test or a genuine incident. Keep unknown outcomes as `unknown`.
Avoid repeated failed sign-ins that could lock an account; review the local
account policy before running any such scenario.

A future label file can use this header:

```csv
scenario_id,start_utc,end_utc,label,attack_family,evidence_ref
```

Use `normal`, `attack`, or `unknown` for `label`; use `R2L` for
`attack_family` only when evidence supports it. A packet capture or a model
prediction alone does not prove whether a login was unauthorized.

## 6. Hand off the minimum useful sample

Once preflight and event checks succeed, share the audit settings, event
counts, capture metadata, Zeek log row counts, and a few redacted sample records. Keep raw Security
events and packet files in `C:\Users\<WindowsUser>\nids-pilot`. The developer will
define the Windows event/packet schema, build new preprocessing and model code,
then return to steps 7–10 in the
[full live IDS manual](live_ids_manual.md). The model will remain in observation
mode until held-out and pilot results meet the targets in the scope record.

## Sources

1. [Microsoft: audit logon events](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/auditing/basic-audit-logon-events)
2. [Microsoft: PktMon start options](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/pktmon-start)
3. [Microsoft: PktMon overview](https://learn.microsoft.com/en-us/windows-server/networking/technologies/pktmon/pktmon)
4. [Microsoft: successful logon event 4624](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/auditing/event-4624)
5. [Microsoft: failed logon event 4625](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/auditing/event-4625)
6. [Microsoft: PktMon ETL to PCAPNG conversion](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/pktmon-etl2pcap)
7. [Zeek: quick start and offline PCAP analysis](https://docs.zeek.org/en/v8.0.4/quickstart.html)
8. [Microsoft: WSL networking modes](https://learn.microsoft.com/windows/wsl/networking)
