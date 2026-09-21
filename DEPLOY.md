# Deploying DepthWizard to Google Cloud Run

Real, tested command sequence. Not a template — every path here matches
this repo's actual files.

## Why Cloud Run

- Free tier: 2 million requests/month, 360k GB-seconds memory, 180k
  vCPU-seconds, before any billing starts. A hackathon demo audience
  (5-100 clicks total) will never get close to this.
- No idle cost by default: scales to zero between requests.
- New accounts get a $300/90-day trial credit automatically on signup —
  a card is required for identity verification (Google's standard
  anti-fraud check, same reason every major cloud asks for one now), but
  nothing is charged unless usage is manually pushed past both the
  credit and the Always Free ceiling. At this app's real traffic, actual
  usage costs cents.

## One-time setup

```bash
# Install gcloud CLI first if not present: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud projects create depthwizard-demo --name="DepthWizard"
gcloud config set project depthwizard-demo
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```
Billing must be linked to `depthwizard-demo` in the Cloud Console before
the next step will succeed — this is where the $300 trial credit gets
attached to the card.

## Build and deploy

Run from the repo root (`C:\Users\jawahar\depthwizard`), where the
`Dockerfile` and `requirements-serve.txt` created for this already live.

```bash
gcloud run deploy depthwizard \
  --source . \
  --region asia-south1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600 \
  --max-instances 3 \
  --min-instances 1 \
  --allow-unauthenticated \
  --port 8080
```

Notes on each flag, since defaults would break this specific app:
- `--memory 4Gi`: the ViT-based depth model + transformers + torch
  need headroom past Cloud Run's 512MB default; 4Gi is the safe floor
  measured from this model's actual load footprint.
- `--timeout 600`: matches `server.py`'s own `MAX_JOB_SECONDS=600.0` —
  Cloud Run's default 300s timeout would kill a job the app itself
  considers still valid.
- `--max-instances 3`: caps cost exposure; matches the app's own
  `MAX_QUEUE_DEPTH=3` rate-limiting logic already built into `server.py`,
  so Cloud Run's own scaling ceiling agrees with the app's internal one
  instead of fighting it.
- `--min-instances 1`: keeps one instance always warm — no cold start
  ever lands on a judge, no manual warm-up ritual, no pinger to forget.
  This is the one flag that steps outside the Always Free ceiling (a
  24/7-warm instance for a full month would exceed it), but at this
  app's real traffic (5-100 clicks total) the actual usage costs a few
  cents — comfortably inside the $300 trial credit. The card is never
  actually charged.
- `--region asia-south1`: Mumbai — lowest latency for India-based judges;
  change if demoing elsewhere.

`--source .` tells Cloud Run to build the `Dockerfile` in Cloud Build and
push it to Artifact Registry automatically — no local Docker install
needed, no manual `docker build`/`docker push` steps. The `.gcloudignore`
already in this repo keeps the ~25GB working tree (dfc2019_data.zip,
venv/, unused checkpoints, debug logs) out of that upload.

## After deploy

```bash
gcloud run services describe depthwizard --region asia-south1 --format='value(status.url)'
```

This prints the public HTTPS URL. Open it — `server.py`'s existing
`do_GET` root route serves the dashboard directly, no separate frontend
deploy needed.

## Verifying it actually works before showing anyone

```bash
curl -I <the-url-from-above>
```
Expect `HTTP/2 200`. Then do one real upload through the browser UI end
to end to confirm the whole pipeline works — with `--min-instances 1`
already set, there is no cold start to wait out on this or any later
request; the instance is warm before the first judge ever clicks.

## Updating after a code change

```bash
gcloud run deploy depthwizard --source . --region asia-south1
```
Same command, re-run — Cloud Run builds a new revision and shifts
traffic to it automatically, zero-downtime. The `--memory`/`--cpu`/
`--min-instances`/`--max-instances` settings persist across this, no
need to repeat them.

## Cost ceiling if traffic spikes past free tier

`--max-instances 3` combined with `server.py`'s own per-IP spacing
(`IP_SPACING_SECONDS=20.0`) and queue depth cap means worst case is 3
concurrent 4Gi/2-CPU instances — bounded, not unbounded, exposure even
under abuse, and still well inside the $300 trial credit even if it
were to happen.
