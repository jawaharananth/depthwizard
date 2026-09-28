# DepthWizard on Azure (Azure for Students)

One Ubuntu VM in Central India runs the same server as the laptop, inside
Docker, behind Caddy (automatic HTTPS). The seven showcase places load
instantly; uploads build on the VM's CPU (no Intel iGPU in the cloud), so a
typical 1 km upload takes roughly 3-5 minutes there instead of ~1.5 on the laptop.

## One-time: claim the credit
1. education.github.com/pack -> **Microsoft Azure** -> **Get access** -> **Start free**.
2. Sign in, verify as a student (college e-mail or GitHub), phone code. No card.
3. portal.azure.com -> **Subscriptions** shows **Azure for Students — Active**.
4. Cost Management -> Budgets -> **+ Add**: $80, alert at 80 % to your e-mail.

## Deploy (about 30-45 minutes the first time)
```powershell
az login                                   # opens the browser; sign in once
powershell -File deploy\azure\deploy.ps1   # everything else
```
The script prints `https://depthwizard-xxxxxx.centralindia.cloudapp.azure.com`.

It tries VM sizes in order (B4ms, D4s_v5, D4as_v5, B2ms) and uses the first
your student quota allows. Re-running it updates the site with the current code.

## Save credit
A 4-vCPU VM costs roughly US$3-4 per day while running.
```powershell
az vm deallocate -g depthwizard-rg -n depthwizard-vm   # stop (no CPU billing)
az vm start      -g depthwizard-rg -n depthwizard-vm   # start again, same address
```
Start it an hour before judging; stop it afterwards.

## What is deliberately not uploaded
The IEEE DFC2019 data (licensed, not redistributable), training tiles, past
uploads, caches and any tokens. `make_bundle.py` enforces this.

## Public-server settings (docker-compose.yml)
- uploads up to 300 MB, one per 20 s per address, at most 3 queued;
- a build that exceeds 45 minutes or goes silent for 2 minutes is stopped and
  retried once (build_worker.py watchdog);
- the app port is internal; only 80/443 (Caddy) are open.
