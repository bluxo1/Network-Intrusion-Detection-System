$ErrorActionPreference = 'Stop'

$status = & pktmon status 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Cannot query PktMon status: $status"
}
if (($status | Out-String) -notmatch 'Packet Monitor is not running') {
    throw 'PktMon may already be running; leave that capture alone.'
}

$captureDir = Join-Path $env:USERPROFILE 'nids-pilot\captures'
New-Item -ItemType Directory -Path $captureDir -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$etl = Join-Path $captureDir "pilot-$stamp.etl"
$pcapng = Join-Path $captureDir "pilot-$stamp.pcapng"

Write-Host "Starting a 60-second NIC capture at $(Get-Date -Format o)"
& pktmon start --capture --comp nics --pkt-size 128 --file-size 64 --log-mode circular --file-name $etl
if ($LASTEXITCODE -ne 0) {
    throw 'PktMon did not start.'
}
try {
    Start-Sleep -Seconds 60
}
finally {
    & pktmon stop
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'PktMon stop failed; check pktmon status and stop it manually.'
    }
}
Write-Host "Stopped capture at $(Get-Date -Format o)"

& pktmon etl2pcap $etl --out $pcapng
if ($LASTEXITCODE -ne 0) {
    throw "PCAPNG conversion failed; ETL remains at $etl"
}
Get-Item -LiteralPath $etl, $pcapng | Select-Object Name, Length
