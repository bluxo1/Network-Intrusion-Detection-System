param(
    [ValidateRange(1, 1440)] [int] $DurationMinutes = 10,
    [ValidateRange(10, 300)] [int] $PollSeconds = 60,
    [ValidateRange(1, 100)] [int] $Threshold = 4,
    [ValidateRange(1, 1440)] [int] $WindowMinutes = 60,
    [ValidateRange(1, 1440)] [int] $CooldownMinutes = 60
)

$ErrorActionPreference = 'Stop'
Get-WinEvent -ListLog Security -ErrorAction Stop | Out-Null
$pilotDir = Join-Path $env:USERPROFILE 'nids-pilot'
New-Item -ItemType Directory -Path $pilotDir -Force | Out-Null
$statePath = Join-Path $pilotDir 'auth_watch_state.json'
$alertPath = Join-Path $pilotDir 'auth_candidate_alerts.jsonl'
$lastAlertAt = $null
$lastSeenRecordId = 0
if (Test-Path -LiteralPath $statePath) {
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ($state.last_alert_at_utc) {
        $lastAlertAt = [datetimeoffset]::Parse($state.last_alert_at_utc)
    }
    if ($state.last_seen_record_id) {
        $lastSeenRecordId = [long]$state.last_seen_record_id
    }
    elseif ($state.last_record_id) {
        $lastSeenRecordId = [long]$state.last_record_id
    }
}

$end = (Get-Date).AddMinutes($DurationMinutes)
Write-Host "Watching Security 4625 for $DurationMinutes minutes; observation only."
do {
    $now = Get-Date
    $filter = @{
        LogName = 'Security'
        Id = 4625
        StartTime = $now.AddMinutes(-$WindowMinutes)
    }
    try {
        $failures = @(Get-WinEvent -FilterHashtable $filter -ErrorAction Stop)
    }
    catch {
        if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound,*') {
            $failures = @()
        }
        else {
            throw
        }
    }
    $count = $failures.Count
    Write-Host "$(Get-Date -Format T): failures in last $WindowMinutes minutes = $count"

    $securityHead = Get-WinEvent -LogName Security -MaxEvents 1 -ErrorAction Stop
    if ($null -ne $securityHead -and [long]$securityHead.RecordId -lt $lastSeenRecordId) {
        throw 'Security log record IDs moved backwards; review the local watch state before continuing.'
    }
    $ordered = @($failures | Sort-Object TimeCreated, RecordId)
    if ($count -gt 0) {
        $latestRecordId = [long](($failures | Sort-Object RecordId -Descending | Select-Object -First 1).RecordId)
    }
    $window = [System.Collections.ArrayList]::new()
    $candidate = $null
    $candidateCount = 0
    foreach ($failure in $ordered) {
        [void]$window.Add($failure)
        while ($window.Count -gt 0 -and
            ($failure.TimeCreated - $window[0].TimeCreated).TotalMinutes -ge $WindowMinutes) {
            $window.RemoveAt(0)
        }
        if ([long]$failure.RecordId -le $lastSeenRecordId) { continue }
        $cooldownExpired = ($null -eq $lastAlertAt) -or
            (([datetimeoffset]::Now - $lastAlertAt).TotalMinutes -ge $CooldownMinutes)
        if ($null -eq $candidate -and $window.Count -ge $Threshold -and $cooldownExpired) {
            $candidate = $failure
            $candidateCount = $window.Count
        }
    }
    if ($null -ne $candidate) {
        $alertAt = [datetimeoffset]::Now
        $alert = [ordered]@{
            schema_version = 'auth-alert-v1'
            event_time_utc = $candidate.TimeCreated.ToUniversalTime().ToString('o')
            alert_time_utc = $alertAt.ToUniversalTime().ToString('o')
            newest_record_id = [long]$candidate.RecordId
            failures_in_window = $candidateCount
            window_minutes = $WindowMinutes
            rule_threshold = $Threshold
            mode = 'observation_only'
        }
        ($alert | ConvertTo-Json -Compress) | Add-Content -LiteralPath $alertPath
        $lastAlertAt = $alertAt
        Write-Warning "Candidate alert: $candidateCount failures in $WindowMinutes minutes."
    }
    if ($count -gt 0 -and $latestRecordId -gt $lastSeenRecordId) {
        $lastSeenRecordId = $latestRecordId
        @{ last_alert_at_utc = if ($null -eq $lastAlertAt) { $null } else { $lastAlertAt.ToUniversalTime().ToString('o') }; last_seen_record_id = $lastSeenRecordId } |
            ConvertTo-Json -Compress | Set-Content -LiteralPath $statePath
    }

    if ((Get-Date) -lt $end) {
        Start-Sleep -Seconds $PollSeconds
    }
} while ((Get-Date) -lt $end)

Write-Host 'Observation complete.'
