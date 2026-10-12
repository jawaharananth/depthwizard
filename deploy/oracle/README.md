# Deploy DepthWizard on Oracle Cloud (free) — from scratch

This puts the full DepthWizard app online on your own domain: the dashboard, the seven showcase
places, and **image upload → 3D model**. It runs on Oracle Cloud's *Always Free* server, so it
costs nothing. You only need a laptop with a web browser and a terminal; nothing else from the
original laptop is needed.

Total time: about 45 minutes, most of it waiting.

---

## What you need before starting

- An email address and a debit or credit card (Oracle uses the card only to verify you; Always Free
  resources are never charged).
- Your domain name, e.g. `depthwizardvantage.tech`, and the login for the site where you bought it.

---

## Step 1 — Create an SSH key on your laptop (2 min)

The key lets you log in to the server. Open **PowerShell** (Windows) or **Terminal** (Mac/Linux) and run:

```
ssh-keygen -t ed25519 -f ~/.ssh/oracle_dw -N ""
```

Then print the public half, and copy the whole line it shows:

```
cat ~/.ssh/oracle_dw.pub
```

(On Windows PowerShell, if `~` does not work, use `$HOME\.ssh\oracle_dw` instead.)

---

## Step 2 — Create the Oracle account (10 min)

1. Go to **https://signup.cloud.oracle.com** and sign up.
2. For **Home Region**, choose **India West (Mumbai)** or **India South (Hyderabad)**. It cannot be
   changed later.
3. Add the card when asked, and finish the sign-up. Wait for the "account ready" email.

---

## Step 3 — Create the free server (5 min)

1. In the Oracle console, open **☰ Menu → Compute → Instances → Create instance**.
2. **Name:** `depthwizard`.
3. **Image and shape:**
   - Click **Change image**, choose **Canonical Ubuntu 22.04**.
   - Click **Change shape**, choose **Ampere → VM.Standard.A1.Flex**, set **4 OCPUs** and **24 GB memory**.
     It should say *Always Free-eligible*.
4. **Networking:** keep the defaults, and make sure **Assign a public IPv4 address** is on.
5. **Add SSH keys:** choose **Paste public keys** and paste the line you copied in Step 1.
6. **Boot volume:** tick *Specify a custom boot volume size* and enter **100** GB (still free).
7. Click **Create**. When it turns green (Running), copy its **Public IP address**.

If you see **"Out of capacity"**, pick a different *Availability domain* on the same page and try
again, or retry in a few hours. This is common for the free ARM servers.

---

## Step 4 — Open the web ports in Oracle (2 min)

1. On the instance page, click the **Subnet** link, then **Default Security List for …**.
2. Click **Add Ingress Rules** and enter:
   - Source CIDR: `0.0.0.0/0`
   - IP Protocol: `TCP`
   - Destination Port Range: `80,443`
3. Click **Add Ingress Rules**.

---

## Step 5 — Point your domain at the server (5 min, then wait)

At the site where you manage the domain's DNS (your registrar, or Cloudflare if you moved it there),
add these two records, using the IP from Step 3:

| Type | Name | Value |
|---|---|---|
| A | `@` | your server's public IP |
| A | `www` | your server's public IP |

Delete any older `A` or `CNAME` record for `@` or `www` first.
**If the domain is on Cloudflare,** set both records to **DNS only** (grey cloud), at least for the
first start, so the server can get its HTTPS certificate.

Check it has taken effect (can take from minutes up to a few hours):

```
nslookup yourdomain.tech
```

It should answer with your server's IP.

---

## Step 6 — Install DepthWizard on the server (1 command, ~20 min)

Log in to the server from your laptop (replace the IP):

```
ssh -i ~/.ssh/oracle_dw ubuntu@YOUR_SERVER_IP
```

Type `yes` if it asks about the fingerprint. Then, on the server, run this one line
(replace the domain with yours):

```
curl -fsSL https://raw.githubusercontent.com/jawaharananth/depthwizard/main/deploy/oracle/setup.sh | bash -s -- yourdomain.tech
```

It opens the firewall, installs Docker, downloads the code, the model and the seven showcase places,
builds the app and starts it with HTTPS. The first build takes 10–20 minutes.

---

## Step 7 — Check it works

Open **https://yourdomain.tech** in a browser. You should see the dashboard.

- Click a place under **Places** → the 3D model opens.
- Drop a satellite image into **Reconstruct an image** → a live progress log appears, and the 3D
  model opens when it finishes (about 6–10 minutes for a 1 km scene).

---

## Everyday commands (on the server)

```
cd ~/depthwizard/deploy/oracle
sudo docker compose ps                 # is it running?
sudo docker compose logs -f app        # live logs (Ctrl+C to stop watching)
sudo docker compose restart            # restart
bash ~/depthwizard/deploy/oracle/setup.sh yourdomain.tech   # pull the latest code and rebuild
```

The app restarts on its own if the server reboots.

---

## If something goes wrong

| Problem | Fix |
|---|---|
| Browser says "site can't be reached" | Check Step 4 (ports 80, 443) and that `nslookup` shows your IP (Step 5). |
| "Your connection is not private" | The certificate is still being issued; wait 2 minutes. If it persists, the domain does not point at the server yet, or Cloudflare is on orange cloud. |
| Upload says the file is too large | The public limit is 95 MB. Crop the image or save it as a compressed GeoTIFF. |
| `setup.sh` stops during the download | Run the same command again; it continues. |
| "Out of capacity" when creating the server | Try another availability domain, or retry later. |

Limits on this server: uploads up to 95 MB, images up to 4096 px on the long side, 45 minutes per build.
