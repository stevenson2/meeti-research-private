# download_worker.ps1 - resumable MIMIC-IV-ECG downloader (PowerShell 5.1 compatible)
# Plan requires the $UrlsFile parameter (case-sensitive) and $urlList variable.
param(
  [Parameter(Mandatory=$true)][string]$UrlsFile,
  [Parameter(Mandatory=$true)][string]$Dest,
  [Parameter(Mandatory=$true)][string]$Log
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if (-not (Test-Path -LiteralPath $UrlsFile)) { throw "UrlsFile not found: $UrlsFile" }
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$urlList = @(Get-Content -LiteralPath $UrlsFile | Where-Object { $_ -match '^https?://' })
if ($urlList.Count -eq 0) { throw "no http(s) urls in $UrlsFile" }

$failedFile = Join-Path $Dest 'failed_urls.txt'
$sw = [Diagnostics.Stopwatch]::StartNew()
Add-Content -LiteralPath $Log -Value ("START urls={0} dest={1}" -f $urlList.Count,$Dest)

$done = 0; $ok = 0; $skip = 0; $fail = 0
foreach ($u in $urlList) {
  $name = [System.IO.Path]::GetFileName($u)
  $out = Join-Path $Dest $name
  $done++
  $finalOk = $false
  if (Test-Path -LiteralPath $out) {
    $len = (Get-Item -LiteralPath $out).Length
    if ($name -like '*.dat') { $finalOk = ($len -eq 120000) } else { $finalOk = ($len -ge 100) }
  }
  if ($finalOk) { $skip++; continue }

  $okThis = $false
  for ($r = 0; $r -lt 4 -and -not $okThis; $r++) {
    $tmp = "$out.part"
    try {
      Invoke-WebRequest -Uri $u -OutFile $tmp -TimeoutSec 90 -ErrorAction Stop
      $len = (Get-Item -LiteralPath $tmp).Length
      if ($name -like '*.dat' -and $len -ne 120000) { throw "bad dat size $len" }
      if ($name -like '*.hea' -and $len -lt 100) { throw "bad hea size $len" }
      Move-Item -LiteralPath $tmp -Destination $out -Force
      $okThis = $true
    } catch {
      Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
      Start-Sleep -Milliseconds (250 * ($r + 1))
    }
  }
  if ($okThis) { $ok++ } else {
    $fail++
    Add-Content -LiteralPath $failedFile -Value $u
  }
  if (($done % 100) -eq 0) {
    $el = [int]$sw.Elapsed.TotalSeconds
    Add-Content -LiteralPath $Log -Value ("{0} done={1}/{2} ok={3} skip={4} fail={5} elapsed={6}s" -f (Get-Date -Format HH:mm:ss),$done,$urlList.Count,$ok,$skip,$fail,$el)
  }
}
$sw.Stop()
$msg = "FINISHED urls={0} ok={1} skip={2} fail={3} elapsed={4}s" -f $urlList.Count,$ok,$skip,$fail,[int]$sw.Elapsed.TotalSeconds
Add-Content -LiteralPath $Log -Value $msg
Write-Output $msg
