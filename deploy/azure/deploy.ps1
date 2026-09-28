# DepthWizard -> one Azure VM (Azure for Students), end to end.
#
#   1. az login                      (once; opens your browser)
#   2. powershell -File deploy\azure\deploy.ps1
#
# Creates a resource group and an Ubuntu VM in Central India, opens 80/443,
# copies the app bundle, builds the container on the VM and starts it behind
# HTTPS. Prints the web address at the end. Safe to re-run: existing
# resources are reused and the app is rebuilt with the current bundle.
#
# Stop paying while nobody is using it:
#   az vm deallocate -g depthwizard-rg -n depthwizard-vm     (stops billing for CPU)
#   az vm start      -g depthwizard-rg -n depthwizard-vm     (same address comes back)

param(
    [string]$Group = "depthwizard-rg",
    [string]$Name = "depthwizard-vm",
    [string]$Location = "centralindia",
    [string[]]$Sizes = @("Standard_B4ms", "Standard_D4s_v5", "Standard_D4as_v5", "Standard_B2ms"),
    [string]$Label = ""
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path

function Step($m) { Write-Host "`n== $m" -ForegroundColor Cyan }

Step "checking the Azure sign-in"
$acct = az account show --query "{name:name, state:state}" -o json 2>$null | ConvertFrom-Json
if (-not $acct) { throw "Not signed in. Run: az login" }
Write-Host "subscription: $($acct.name) ($($acct.state))"

if (-not $Label) {
    $h = [System.BitConverter]::ToString((New-Object Security.Cryptography.SHA1Managed).ComputeHash([Text.Encoding]::UTF8.GetBytes($acct.name))).Replace("-", "").ToLower()
    $Label = "depthwizard-" + $h.Substring(0, 6)
}
$Domain = "$Label.$Location.cloudapp.azure.com"

Step "building the app bundle"
& "$Root\venv\Scripts\python.exe" "$Root\deploy\azure\make_bundle.py"
if ($LASTEXITCODE -ne 0) { throw "bundle failed" }
$Bundle = "$Root\dist\depthwizard_bundle.tar.gz"

Step "resource group $Group in $Location"
az group create -n $Group -l $Location -o none

$exists = az vm show -g $Group -n $Name --query name -o tsv 2>$null
if (-not $exists) {
    $made = $false
    foreach ($size in $Sizes) {
        Step "creating VM $Name ($size) -- a few minutes"
        az vm create -g $Group -n $Name --image Ubuntu2204 --size $size `
            --admin-username azureuser --generate-ssh-keys `
            --os-disk-size-gb 64 --public-ip-sku Standard `
            --public-ip-address-dns-name $Label `
            --custom-data "$Root\deploy\azure\cloud-init.yaml" -o none 2>&1 | Tee-Object -Variable out | Out-Null
        if ($LASTEXITCODE -eq 0) { $made = $true; Write-Host "VM created: $size"; break }
        Write-Host "  $size not available on this subscription -- trying the next size"
        az vm delete -g $Group -n $Name --yes -o none 2>$null
    }
    if (-not $made) { throw "no VM size could be created -- see Quotas in the portal" }
    az vm open-port -g $Group -n $Name --port 80,443 --priority 900 -o none
} else {
    Write-Host "VM exists -- starting it if stopped"
    az vm start -g $Group -n $Name -o none
}

$ip = az vm show -d -g $Group -n $Name --query publicIps -o tsv
Write-Host "public IP $ip, address https://$Domain"
$ssh = "azureuser@$ip"
$opts = @("-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30")

Step "waiting for the VM's first-boot setup (Docker)"
ssh @opts $ssh "cloud-init status --wait >/dev/null; docker --version"

Step "copying the app bundle (about 1-2 GB; depends on your upload speed)"
scp @opts $Bundle "${ssh}:/opt/depthwizard/bundle.tar.gz"

Step "building and starting on the VM (the first build takes 10-20 minutes)"
ssh @opts $ssh "set -e; cd /opt/depthwizard; tar xzf bundle.tar.gz; cd deploy/azure; DW_DOMAIN=$Domain sudo -E docker compose up -d --build; sudo docker compose ps"

Step "waiting for the site to answer"
for ($i = 0; $i -lt 60; $i++) {
    try { $r = Invoke-WebRequest -UseBasicParsing "https://$Domain/health" -TimeoutSec 10; if ($r.StatusCode -eq 200) { break } } catch {}
    Start-Sleep -Seconds 10
}
Write-Host "`nDepthWizard is live:  https://$Domain" -ForegroundColor Green
Write-Host "Stop billing when idle:  az vm deallocate -g $Group -n $Name"
