"""Hand the upload server the awkward files a judge might, and record what happens.

    python server.py --port 8007 &
    python scripts/smoke_uploads.py http://127.0.0.1:8007 <folder of test files>

For each file: HTTP status of the upload, then (if accepted) the job's final
state, the product written (DSM / rDSM) and the message shown to the user.
A refusal is a pass when its message says what to do instead; a crash or a
silent wrong product is a fail.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request


def post(base, path):
    data = open(path, "rb").read()
    req = urllib.request.Request(f"{base}/api/upload?name={os.path.basename(path)}", data=data,
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {"error": e.reason}


def wait(base, job, limit_s=1500):
    t0 = time.time()
    while time.time() - t0 < limit_s:
        with urllib.request.urlopen(f"{base}/api/status/{job}", timeout=30) as r:
            s = json.load(r)
        if s.get("state") in ("done", "error"):
            s["seconds"] = round(time.time() - t0)
            return s
        time.sleep(5)
    return {"state": "timeout"}


def main(base, folder):
    rows = []
    for f in sorted(os.listdir(folder)):
        path = os.path.join(folder, f)
        code, body = post(base, path)
        row = {"file": f, "upload_http": code}
        if code == 200 and "job" in body:
            s = wait(base, body["job"])
            row.update(state=s.get("state"), message=s.get("message"), seconds=s.get("seconds"))
            try:
                with urllib.request.urlopen(f"{base}/api/dsm?job={body['job']}", timeout=60) as r:
                    cd = r.headers.get("Content-Disposition", "")
                    row["product"] = cd.split("filename=")[-1].strip('"')
            except urllib.error.HTTPError as e:
                row["product"] = f"none ({e.code})"
            time.sleep(21)           # the server spaces uploads from one address
        else:
            row["message"] = body.get("error")
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
