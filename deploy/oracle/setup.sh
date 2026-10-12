#!/usr/bin/env bash
# DepthWizard on an Oracle Cloud "Always Free" Ubuntu VM (Ampere A1, ARM) -- one command.
#
#   curl -fsSL https://raw.githubusercontent.com/jawaharananth/depthwizard/main/deploy/oracle/setup.sh | bash -s -- yourdomain.tech
#
# What it does, in order:
#   1. opens ports 80/443 in the VM's own firewall (Oracle Ubuntu images block them by default)
#   2. installs Docker
#   3. clones the repo (or updates it)
#   4. downloads the model weights and the 7 showcase scenes from Hugging Face (not stored in git)
#   5. builds the app and starts it behind Caddy, which gets the HTTPS certificate for your domain
# Safe to re-run: it updates the code and restarts.
set -euo pipefail

DOMAIN="${1:-}"
if [ -z "$DOMAIN" ]; then
  echo "usage: bash setup.sh yourdomain.tech   (the domain's A record must already point at this VM)"; exit 1
fi
REPO_URL="https://github.com/jawaharananth/depthwizard.git"
ASSETS_REPO="jawahar173/depthwizard-assets"
DIR="$HOME/depthwizard"

echo "== 1/5 firewall: open 80 and 443"
for p in 80 443; do
  sudo iptables -C INPUT -p tcp --dport $p -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 6 -p tcp --dport $p -j ACCEPT
done
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent git python3-venv curl
sudo netfilter-persistent save

echo "== 2/5 Docker"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi

echo "== 3/5 code"
if [ -d "$DIR/.git" ]; then git -C "$DIR" pull --ff-only; else git clone --depth 1 "$REPO_URL" "$DIR"; fi

echo "== 4/5 model weights + showcase scenes (about 1.8 GB)"
python3 -m venv "$HOME/.hfenv"
"$HOME/.hfenv/bin/pip" install -q --upgrade huggingface_hub
"$HOME/.hfenv/bin/python" - "$DIR" "$ASSETS_REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[2], repo_type="model", local_dir=sys.argv[1])
print("assets ready")
PY

echo "== 5/5 build and start (first build takes 10-20 minutes)"
cd "$DIR/deploy/oracle"
sudo DW_DOMAIN="$DOMAIN" docker compose up -d --build
echo
echo "Done. Open https://$DOMAIN  (the certificate can take a minute on first start)."
echo "Logs:   cd $DIR/deploy/oracle && sudo docker compose logs -f app"
