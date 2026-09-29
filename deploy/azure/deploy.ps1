# DepthWizard -> one Azure VM (Azure for Students), end to end.
#
#   1. az login                      (once; opens your browser)
#   2. powershell -ExecutionPolicy Bypass -File deploy\azure\deploy.ps1
#
# Creates a resource group and an Ubuntu VM (India first, other regions if the
# student subscription does not allow them), opens 80/443, copies the app
# bundle, builds the container on the VM and starts it behind HTTPS. Prints
# the web address at the end. Safe to re-run: existing resources are reused
# and the app is rebuilt with the current bundle.
#
# Stop paying while nobody is using it:
#   az vm deallocate -g depthwizard-rg -n depthwizard-vm     (stops billing for CPU)
#   az vm start      -g depthwizard-rg -n depthwizard-vm     (same address comes back)

param(
    [string]$Group = "depthwizard-rg",
    [string]$Name = "depthwizard-vm",
    # Azure for Students allows only some regions; the first that accepts the
    # VM is used (India first: lowest latency for the judges)
    [string[]]$Locations = @("centralindia", "southindia", "westindia", "southeastasia", "eastus", "westeurope"),
    [string[]]$Sizes = @("Standard_B4ms", "Standard_D4s_v5", "Standard_D4as_v5", "Standard_B2ms"),
    [string]$Label = "",
    [switch]$SkipBundle
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path

function Step($m) { Write-Host "`n== $m" -ForegroundColor Cyan }

# the CLI installer updates PATH only for new sessions
if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    $wbin = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin"
    if (Test-Path "$wbin\az.cmd") { $env:Path = "$wbin;$env:Path" }
    else { throw "Azure CLI not found. Install it: winget install Microsoft.AzureCLI" }
}
foreach ($tool in "ssh", "scp") {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool not found: Settings > System > Optional features > add 'OpenSSH Client'"
    }
}

Step "checking the Azure sign-in"
$acct = az account show --query "{name:name, state:state, id:id}" -o json 2>$null | ConvertFrom-Json
if (-not $acct) { throw "Not signed in. Run: az login" }
Write-Host "subscription: $($acct.name) ($($acct.state))"
if ($acct.state -ne "Enabled") { throw "The subscription is not enabled -- activate Azure for Students first" }

if (-not $Label) {
    $h = [System.BitConverter]::ToString((New-Object Security.Cryptography.SHA1Managed).ComputeHash(
        [Text.Encoding]::UTF8.GetBytes($acct.id))).Replace("-", "").ToLower()
    $Label = "depthwizard-" + $h.Substring(0, 6)
}

$Bundle = "$Root\dist\depthwizard_bundle.tar.gz"
if (-not $SkipBundle -or -not (Test-Path $Bundle)) {
    Step "building the app bundle"
    & "$Root\venv\Scripts\python.exe" "$Root\deploy\azure\make_bundle.py"
    if ($LASTEXITCODE -ne 0) { throw "bundle failed" }
}

az provider register --namespace Microsoft.Compute -o none 2>$null
az provider register --namespace Microsoft.Network -o none 2>$null

$exists = az vm show -g $Group -n $Name --query name -o tsv 2>$null
if (-not $exists) {
    $made = $false
    foreach ($loc in $Locations) {
        Step "trying region $loc"
        $rg = az group show -n $Group --query location -o tsv 2>$null
        if (-not $rg) {
            $gout = az group create -n $Group -l $loc -o none 2>&1
            if ($LASTEXITCODE -ne 0) { Write-Host "  resource group refused in ${loc}: $gout"; continue }
        }
        foreach ($size in $Sizes) {
            Write-Host "  creating VM $Name ($size in $loc) -- a few minutes"
            $out = az vm create -g $Group -n $Name -l $loc --image Ubuntu2204 --size $size `
                --admin-username azureuser --generate-ssh-keys `
                --os-disk-size-gb 64 --public-ip-sku Standard `
                --public-ip-address-dns-name $Label `
                --custom-data "$Root\deploy\azure\cloud-init.yaml" -o none 2>&1
            if ($LASTEXITCODE -eq 0) { $made = $true; Write-Host "  VM created: $size in $loc" -ForegroundColor Green; break }
            $why = ($out | Out-String).Trim()
            Write-Host "    not accepted: $(($why -split "`n")[0])"
            az vm delete -g $Group -n $Name --yes -o none 2>$null
            if ($why -match "RequestDisallowedByPolicy|LocationNotAvailable|NotAvailableForSubscription") { break }   # the region, not the size
        }
        if ($made) { break }
    }
    if (-not $made) { throw "No region/size could be created. Portal: Subscriptions > Usage + quotas" }
    az vm open-port -g $Group -n $Name --port 80,443 --priority 900 -o none
} else {
    Write-Host "VM exists -- starting it if stopped"
    az vm start -g $Group -n $Name -o none
}

$vmLoc = az vm show -g $Group -n $Name --query location -o tsv
$Domain = "$Label.$vmLoc.cloudapp.azure.com"
$ip = az vm show -d -g $Group -n $Name --query publicIps -o tsv
Write-Host "public IP $ip, address https://$Domain"
$ssh = "azureuser@$ip"
$opts = @("-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20")

Step "waiting for the VM's first-boot setup (Docker)"
$ok = $false
for ($i = 0; $i -lt 30; $i++) {
    ssh @opts $ssh "cloud-init status --wait >/dev/null 2>&1; docker --version" 2>$null
    if ($LASTEXITCODE -eq 0) { $ok = $true; break }
    Start-Sleep -Seconds 20
}
if (-not $ok) { throw "the VM did not finish installing Docker -- re-run this script in a few minutes" }

Step "copying the app bundle (about 1.8 GB; depends on your upload speed)"
scp @opts $Bundle "${ssh}:/opt/depthwizard/bundle.tar.gz"
if ($LASTEXITCODE -ne 0) { throw "copy failed -- re-run with -SkipBundle to retry the upload" }

Step "building and starting on the VM (the first build takes 10-20 minutes)"
ssh @opts $ssh "set -e; cd /opt/depthwizard; tar xzf bundle.tar.gz; cd deploy/azure; DW_DOMAIN=$Domain sudo -E docker compose up -d --build; sudo docker compose ps"
if ($LASTEXITCODE -ne 0) { throw "the build on the VM failed -- see: ssh $ssh 'cd /opt/depthwizard/deploy/azure; sudo docker compose logs --tail 80'" }

Step "waiting for the site to answer (model load + HTTPS certificate)"
$live = $false
for ($i = 0; $i -lt 60; $i++) {
    try { $r = Invoke-WebRequest -UseBasicParsing "https://$Domain/health" -TimeoutSec 10; if ($r.StatusCode -eq 200) { $live = $true; break } } catch {}
    Start-Sleep -Seconds 10
}
if ($live) {
    Write-Host "`nDepthWizard is live:  https://$Domain" -ForegroundColor Green
} else {
    Write-Host "`nNot answering yet at https://$Domain -- check: ssh $ssh 'cd /opt/depthwizard/deploy/azure; sudo docker compose logs --tail 80'" -ForegroundColor Yellow
}
Write-Host "Stop billing when idle:  az vm deallocate -g $Group -n $Name"
