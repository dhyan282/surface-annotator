param(
    [string]$HFUsername,
    [string]$SpaceName,
    [string]$HFToken
)

if (-not $HFUsername) {
    $HFUsername = Read-Host "Hugging Face username"
}
if (-not $SpaceName) {
    $SpaceName = Read-Host "Space name (URL will be https://huggingface.co/spaces/${HFUsername}/${SpaceName})"
}
if (-not $SpaceName) {
    $SpaceName = "surface-auto-annotator"
}
if (-not $HFToken) {
    $secure = Read-Host "HF write token (hf_xxx...)" -AsSecureString
    $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    $HFToken = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
    [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($BSTR)
}

if (-not $HFUsername -or -not $SpaceName -or -not $HFToken) {
    Write-Error "username, space name, and token are all required."
    exit 1
}

$RepoDir = (Resolve-Path .).Path
Write-Host ""
Write-Host "==> Target: https://huggingface.co/spaces/${HFUsername}/${SpaceName}"
Write-Host "==> Repo:   $RepoDir"
Write-Host ""

if (-not (Test-Path ".git")) {
    Write-Error "$RepoDir is not a git repo. Run 'git init' first."
    exit 1
}

$branch = git rev-parse --abbrev-ref HEAD
if (-not $branch) {
    Write-Error "No commits yet. Create an initial commit first."
    exit 1
}

Write-Host "==> Creating Space (idempotent: skips if it already exists)..."
$body = @{
    name = $SpaceName
    sdk = "docker"
    private = $false
    hardware = "cpu-basic"
    storage = "small"
} | ConvertTo-Json

try {
    $req = [System.Net.HttpWebRequest]::Create("https://huggingface.co/api/spaces/${HFUsername}/${SpaceName}")
    $req.Method = "POST"
    $req.Headers.Add("Authorization", "Bearer ${HFToken}")
    $req.ContentType = "application/json"
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
    $req.ContentLength = $bytes.Length
    $stream = $req.GetRequestStream()
    $stream.Write($bytes, 0, $bytes.Length)
    $stream.Close()
    $resp = $req.GetResponse()
    $respStream = $resp.GetResponseStream()
    $reader = New-Object System.IO.StreamReader($respStream)
    $respText = $reader.ReadToEnd()
    $reader.Close()
    Write-Host "    Space created: $($resp.StatusCode) $respText"
} catch [System.Net.WebException] {
    if ($_.Exception.Response.StatusCode -eq 409) {
        Write-Host "    Space already exists -- continuing."
    } else {
        Write-Error "HTTP error creating space: $($_.Exception.Message)"
        exit 1
    }
}

$REMOTE = "hf"
$REMOTE_URL = "https://${HFUsername}:${HFToken}@huggingface.co/spaces/${HFUsername}/${SpaceName}"

$existing = git remote get-url $REMOTE 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "==> Remote '$REMOTE' already exists -- updating URL."
    git remote set-url $REMOTE $REMOTE_URL
} else {
    Write-Host "==> Adding remote '$REMOTE' -> $REMOTE_URL"
    git remote add $REMOTE $REMOTE_URL
}

Write-Host "==> Pushing to Space (this can take a few minutes for the first build)..."
if ($branch -ne "main") {
    Write-Host "==> Renaming current branch '${branch}' -> 'main' for HF."
    git branch -M main
}

try {
    git push --force $REMOTE main
} finally {
    git remote set-url $REMOTE "https://huggingface.co/spaces/${HFUsername}/${SpaceName}"
}

Write-Host ""
Write-Host "==> Done! Your Space will start building now."
Write-Host "    URL:      https://huggingface.co/spaces/${HFUsername}/${SpaceName}"
Write-Host "    Logs:     https://huggingface.co/spaces/${HFUsername}/${SpaceName}/logs"
Write-Host "    It usually takes 3-5 minutes to build the Docker image and start the app."
Write-Host "    The first request will take ~30s while SegFormer downloads (~44 MB)."