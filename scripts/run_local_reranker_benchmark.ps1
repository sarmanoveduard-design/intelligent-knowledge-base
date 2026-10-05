[CmdletBinding()]
param(
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Device = 'auto',
    [ValidateRange(1, 1024)][int]$BatchSize = 4,
    [ValidateRange(8, 8192)][int]$MaxLength = 512,
    [string]$Revision = 'main'
)
$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$settings = @{
    BGE_RERANKER_DEVICE = $Device; BGE_RERANKER_BATCH_SIZE = "$BatchSize"
    BGE_RERANKER_MAX_LENGTH = "$MaxLength"; BGE_RERANKER_REVISION = $Revision
}
$savedSettings = @{}
$emptyEnvironmentFile = $null
$failureMessage = 'Reranker preparation failed.'

function Invoke-DockerQuiet {
    param([string[]]$Arguments)
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $captured = @(& docker @Arguments 2>$null)
        $exitCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    return [pscustomobject]@{ Output = $captured; ExitCode = $exitCode }
}

Push-Location $repositoryRoot
try {
    $failureMessage = 'Invalid model revision; use main or a full model commit SHA.'
    if ($Revision -ne 'main' -and $Revision -notmatch '^[0-9a-fA-F]{40}$') { throw $failureMessage }
    $failureMessage = 'Prepared corpus.json and gold.json snapshots are required.'
    if (-not (Test-Path -LiteralPath 'data/benchmark/corpus.json') -or -not (Test-Path -LiteralPath 'data/benchmark/gold.json')) {
        throw $failureMessage
    }
    $failureMessage = 'Git SHA/dirty metadata could not be determined.'
    $commitOutput = & git rev-parse HEAD 2>$null
    if ($LASTEXITCODE -ne 0) { throw $failureMessage }
    $commitSha = ($commitOutput | Out-String).Trim().ToLowerInvariant()
    if ($commitSha -notmatch '^(?:[0-9a-f]{40}|[0-9a-f]{64})$') { throw $failureMessage }
    $dirtyOutput = & git status --porcelain 2>$null
    if ($LASTEXITCODE -ne 0) { throw $failureMessage }
    $settings.BENCHMARK_CODE_COMMIT_SHA = $commitSha
    $settings.BENCHMARK_CODE_DIRTY = if ($dirtyOutput) { 'true' } else { 'false' }
    foreach ($key in $settings.Keys) {
        $savedSettings[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, [string]$settings[$key], 'Process')
    }
    $failureMessage = 'Docker is unavailable; start Docker Desktop.'
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw $failureMessage }
    $result = Invoke-DockerQuiet -Arguments @('info', '--format', '{{.ServerVersion}}')
    if ($result.ExitCode -ne 0) { throw $failureMessage }
    $failureMessage = 'Docker Compose is unavailable.'
    $result = Invoke-DockerQuiet -Arguments @('compose', 'version', '--short')
    if ($result.ExitCode -ne 0) { throw $failureMessage }
    # Empty --env-file prevents implicit .env reads. Same first file/directory
    # and process COMPOSE_PROJECT_NAME as the existing Compose project.
    $emptyEnvironmentFile = [System.IO.Path]::GetTempFileName()
    $composeArguments = @('compose', '--env-file', $emptyEnvironmentFile, '--project-directory', $repositoryRoot,
        '-f', 'compose.yaml', '-f', 'compose.gpu.yaml', '-f', 'compose.reranker.yaml')
    # Resolve a custom existing project from this repository's Compose labels,
    # without reading .env or inspecting/mutating unrelated projects.
    if (-not $env:COMPOSE_PROJECT_NAME) {
        $failureMessage = 'Cannot determine the existing Compose project.'
        $result = Invoke-DockerQuiet -Arguments @('ps', '-a', '--filter',
            "label=com.docker.compose.project.working_dir=$repositoryRoot", '--format', '{{.Label "com.docker.compose.project"}}')
        if ($result.ExitCode -ne 0) { throw $failureMessage }
        $projects = @($result.Output | Where-Object { $_ } | Sort-Object -Unique)
        if ($projects.Count -gt 1) {
            $failureMessage = 'Multiple Compose projects use this repository; select COMPOSE_PROJECT_NAME.'
            throw $failureMessage
        }
        if ($projects.Count -eq 1) { $composeArguments += @('--project-name', $projects[0]) }
    }
    if ($Device -ne 'cpu') { $composeArguments += @('-f', 'compose.reranker.gpu.yaml') }
    $failureMessage = 'Cannot inspect project Ollama service.'
    $result = Invoke-DockerQuiet -Arguments ($composeArguments + @('ps', '--status', 'running', '--quiet', 'ollama'))
    if ($result.ExitCode -ne 0) { throw $failureMessage }
    if (-not $result.Output) {
        $failureMessage = 'Project Ollama could not start; check Docker GPU runtime.'
        $result = Invoke-DockerQuiet -Arguments ($composeArguments + @('up', '-d', '--no-recreate', 'ollama'))
        if ($result.ExitCode -ne 0) { throw $failureMessage }
    }
    $failureMessage = 'Reranker runtime build failed; check Docker and dependency downloads.'
    $result = Invoke-DockerQuiet -Arguments ($composeArguments + @('build', 'reranker-benchmark'))
    if ($result.ExitCode -ne 0) { throw $failureMessage }
    $failureMessage = 'Project Ollama is unavailable from reranker runtime; check Compose network/GPU runtime.'
    $runArguments = $composeArguments + @('run', '--rm', '--no-deps', '-T', 'reranker-benchmark')
    $result = Invoke-DockerQuiet -Arguments ($runArguments + @('python', 'scripts/check_reranker_ollama.py'))
    if ($result.ExitCode -eq 2) { $failureMessage = 'BGE-M3 missing in project Ollama; no model was downloaded.' }
    if ($result.ExitCode -ne 0) { throw $failureMessage }
    $corpusHash = (Get-FileHash data/benchmark/corpus.json -Algorithm SHA256).Hash.ToLowerInvariant()
    $goldHash = (Get-FileHash data/benchmark/gold.json -Algorithm SHA256).Hash.ToLowerInvariant()
    $failureMessage = 'Benchmark did not create a report; check model cache, revision and device.'
    $result = Invoke-DockerQuiet -Arguments ($runArguments + @('python', 'scripts/eval_retrieval.py',
        '--corpus', 'data/benchmark/corpus.json', '--gold', 'data/benchmark/gold.json', '--top-k', '10'))
    $benchmarkExit = $result.ExitCode
    $reportLine = $result.Output | Where-Object { $_ -match '^Report: reports/[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}/REPORT\.md$' } | Select-Object -Last 1
    if (-not $reportLine) { throw $failureMessage }
    Write-Output $reportLine
    $failureMessage = 'Report comparison failed; check report files.'
    $result = Invoke-DockerQuiet -Arguments ($runArguments + @('python', 'scripts/compare_retrieval_reports.py',
        '--reports-root', 'reports', '--out', 'reports', '--corpus-hash', $corpusHash, '--gold-hash', $goldHash))
    if ($result.ExitCode -ne 0) { throw $failureMessage }
    Write-Output 'Comparison: reports/COMPARISON.md'
    $failureMessage = 'Benchmark is incomplete; inspect sanitized report errors.'
    if ($benchmarkExit -ne 0) { throw $failureMessage }
    Write-Output 'Benchmark completed: dense BGE-M3 -> Top-20 -> BGE reranker -> Top-10.'
} catch {
    # Forward only controlled stage messages, never native/model output.
    throw $failureMessage
} finally {
    foreach ($key in $savedSettings.Keys) {
        [Environment]::SetEnvironmentVariable($key, $savedSettings[$key], 'Process')
    }
    if ($emptyEnvironmentFile) { Remove-Item -LiteralPath $emptyEnvironmentFile -Force -ErrorAction SilentlyContinue }
    Pop-Location
}
