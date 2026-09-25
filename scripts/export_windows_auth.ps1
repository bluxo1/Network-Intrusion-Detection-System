param(
    [ValidateRange(1, 30)] [int] $Days = 7,
    [string] $OutputPath = (Join-Path $env:USERPROFILE 'nids-pilot\auth_failures_7d.csv')
)

$ErrorActionPreference = 'Stop'

# Fail visibly if this shell cannot read Security instead of exporting a false zero.
Get-WinEvent -ListLog Security -ErrorAction Stop | Out-Null
$filter = @{
    LogName = 'Security'
    Id = 4625
    StartTime = (Get-Date).AddDays(-$Days)
}
try {
    $events = @(Get-WinEvent -FilterHashtable $filter -ErrorAction Stop)
}
catch {
    if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound,*') {
        $events = @()
    }
    else {
        throw
    }
}
$rows = @(
    foreach ($entry in $events) {
        [pscustomobject]@{
            time_local = $entry.TimeCreated.ToString('o')
            event_id = $entry.Id
            record_id = $entry.RecordId
            logon_type = $entry.Properties[10].Value
            status_signed = $entry.Properties[7].Value
            substatus_signed = $entry.Properties[9].Value
        }
    }
)

$parent = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Path $parent -Force | Out-Null
$tempPath = Join-Path $parent ('.auth-export-' + [guid]::NewGuid().ToString('N') + '.tmp')
if ($rows.Count -eq 0) {
    'time_local,event_id,record_id,logon_type,status_signed,substatus_signed' |
        Set-Content -LiteralPath $tempPath -Encoding UTF8
}
else {
    $rows | Export-Csv -LiteralPath $tempPath -NoTypeInformation -Encoding UTF8
}
Move-Item -LiteralPath $tempPath -Destination $OutputPath -Force
Write-Host "Failed logons in last $Days days: $($rows.Count)"
Write-Host "Saved locally: $OutputPath"
