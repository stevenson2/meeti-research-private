[CmdletBinding()]
param(
    [string]$PathsJson,
    [switch]$InitializeSnapshot
)

$ErrorActionPreference = 'Stop'
$targetRoot = Split-Path -Parent $PSScriptRoot
if (-not $PathsJson) {
    $workspaceRoot = Split-Path -Parent $targetRoot
    $PathsJson = Join-Path $workspaceRoot 'ecg-programme-tju-flex.electron-master\scripts\project_paths.json'
}
$config = Get-Content -LiteralPath $PathsJson -Raw -Encoding UTF8 | ConvertFrom-Json
$sourceRoot = (Resolve-Path -LiteralPath $config.windows.meeti_root).ProviderPath

function Invoke-SourceGit {
    param([string[]]$GitArgs)
    $output = & git -c "safe.directory=$sourceRoot" -c core.autocrlf=false -c core.eol=lf -C $sourceRoot @GitArgs
    if ($LASTEXITCODE -ne 0) { throw 'Source Git command failed.' }
    return $output
}

if (-not (Test-Path -LiteralPath (Join-Path $targetRoot '.git'))) {
    throw 'Initialize this upload repository before syncing.'
}
$dirty = & git -C $targetRoot status --porcelain -- scripts
if ($LASTEXITCODE -ne 0) { throw 'Target Git status failed.' }
if ($dirty) {
    & git -C $targetRoot rev-parse --verify HEAD *> $null
    $hasCommit = $LASTEXITCODE -eq 0
    $hasTrackedChanges = @($dirty | Where-Object { -not $_.StartsWith('?? ') }).Count -gt 0
    if (-not $InitializeSnapshot -or $hasCommit -or $hasTrackedChanges) {
        throw 'Target scripts have uncommitted changes; inspect them before syncing.'
    }
    # Explicit first-export retry: overwrite generated untracked copies, never delete files.
}

$sourceHead = Invoke-SourceGit -GitArgs @('rev-parse', 'HEAD')
$treeLines = @(Invoke-SourceGit -GitArgs @('-c', 'core.quotepath=false', 'ls-tree', '-r', '--format=%(objectmode) %(objectname) %(path)', $sourceHead, '--', 'scripts/'))
$tree = @{}
foreach ($line in $treeLines) {
    if ($line -notmatch '^(100644|100755) ([a-f0-9]{40}) (scripts/.+)$') {
        throw 'Source scripts must be ordinary Git files; unexpected tree entry.'
    }
    $tree[$Matches[3]] = @{ mode = $Matches[1]; blob = $Matches[2] }
}
$paths = @($tree.Keys | Sort-Object)
$allowedExtensions = @('.py', '.ps1', '.sh', '.json', '.md', '.txt')
foreach ($relative in $paths) {
    if (-not $relative.StartsWith('scripts/') -or $relative.Contains('..') -or
        [IO.Path]::GetExtension($relative) -notin $allowedExtensions -or
        $relative -match '(?i)(NEXT_SESSION_PROMPT[^/]*\.md$|(^|/)PLAN_STATE\.json$|(^|/)SESSION_LOG\.md$|(^|/)LOOP_LOG\.md$|(^|/)local_meeti_metadata_manifest\.[^/]+$|(^|/)(credentials|secrets|\.env)([./]|$))') {
        throw "Unapproved script path: $relative"
    }
}
if ($paths.Count -eq 0) { throw 'Source contains no eligible committed scripts.' }

$archivePath = [IO.Path]::GetTempFileName()
try {
    Invoke-SourceGit -GitArgs (@('archive', '--format=tar', "--output=$archivePath", $sourceHead, '--') + $paths) | Out-Null
    & tar -xf $archivePath -C $targetRoot
    if ($LASTEXITCODE -ne 0) { throw 'Archive extraction failed.' }

    $files = foreach ($relative in $paths) {
        $localPath = Join-Path $targetRoot $relative
        $expectedBlob = $tree[$relative].blob
        $actualBlob = & git -C $targetRoot hash-object --no-filters -- $localPath
        if ($LASTEXITCODE -ne 0 -or $actualBlob -ne $expectedBlob) {
            throw "Copied script does not match source Git blob: $relative"
        }
        [ordered]@{
            path = $relative
            source_git_blob = $expectedBlob
            source_git_mode = $tree[$relative].mode
            sha256 = (Get-FileHash -LiteralPath $localPath -Algorithm SHA256).Hash.ToLowerInvariant()
            bytes = (Get-Item -LiteralPath $localPath).Length
        }
    }
    $manifest = [ordered]@{
        schema_version = 1
        exported_at_utc = [DateTime]::UtcNow.ToString('o')
        source_commit = $sourceHead
        scope = 'Committed scripts/ only; no source Git history, runs, data, weights, logs, fulltext, or session state.'
        files = @($files)
    }
    $manifestPath = Join-Path $targetRoot 'docs\source_snapshot.json'
    [IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 6) + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
    Write-Output "Copied $($paths.Count) committed scripts from $sourceHead. No commit or push was performed."
} finally {
    if (Test-Path -LiteralPath $archivePath) { Remove-Item -LiteralPath $archivePath }
}
