// DepthWizard desktop shell.
//
// 1. Opens the bundled offline viewer (dist/): a landing page linking to the
//    pre-built scenes. Scene files (~480 MB) are NOT compiled into the binary;
//    they ship as bundle resources and /scenes/<name>/<file> is served from disk.
// 2. Tries to start the Python processing backend (server.py) bound to
//    127.0.0.1 on a free port. Once it answers, the window switches to its
//    upload dashboard. The backend is killed (whole process tree) on exit.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

const DEV_HOME: &str = r"C:\Users\jawahar\depthwizard";

/// Directories that contain server.py + venv, in priority order.
fn backend_homes(exe_dir: &Path) -> Vec<PathBuf> {
    let mut v = vec![exe_dir.join("backend")];
    if let Ok(h) = std::env::var("DEPTHWIZARD_HOME") {
        v.push(PathBuf::from(h));
    }
    v.push(PathBuf::from(DEV_HOME));
    v
}

fn find_backend(exe_dir: &Path) -> Option<(PathBuf, PathBuf)> {
    backend_homes(exe_dir).into_iter().find_map(|home| {
        if !home.join("server.py").is_file() {
            return None;
        }
        // The installer ships an embedded runtime (backend\python); a
        // developer checkout uses its venv.
        [r"python\python.exe", r"venv\Scripts\python.exe"]
            .iter()
            .map(|p| home.join(p))
            .find(|py| py.is_file())
            .map(|py| (home, py))
    })
}

fn spawn_backend(home: &Path, py: &Path, port: u16) -> std::io::Result<Child> {
    let log = std::fs::File::create(std::env::temp_dir().join("depthwizard-backend.log"))?;
    let mut cmd = Command::new(py);
    cmd.arg("server.py")
        .args(["--host", "127.0.0.1", "--port", &port.to_string()])
        .current_dir(home)
        .env("PYTHONUNBUFFERED", "1")
        .env("DW_SCENES_DIR", home.parent().map(|p| p.join("scenes")).unwrap_or_default())
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd.spawn()
}

fn kill_tree(child: &mut Child) {
    // taskkill /T also stops any pipeline subprocesses server.py started.
    let _ = Command::new("taskkill")
        .args(["/T", "/F", "/PID", &child.id().to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    let _ = child.kill();
    let _ = child.wait();
}

fn mime(p: &Path) -> &'static str {
    match p.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "json" | "geojson" => "application/json",
        "glb" => "model/gltf-binary",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        _ => "application/octet-stream",
    }
}

fn main() {
    let child: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));
    let child_exit = child.clone();

    let app = tauri::Builder::default()
        .setup(move |app| {
            let exe_dir = std::env::current_exe()?.parent().unwrap().to_path_buf();
            let mut scene_roots = vec![];
            if let Ok(r) = app.path().resource_dir() {
                scene_roots.push(r.join("scenes"));
            }
            scene_roots.extend(backend_homes(&exe_dir).into_iter().map(|h| h.join("scenes")));

            let win = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("DepthWizard")
                .inner_size(1440.0, 900.0)
                .on_web_resource_request(move |req, resp| {
                    let path = req.uri().path();
                    let Some(rel) = path.strip_prefix("/scenes/") else { return };
                    let rel = percent_decode(rel);
                    if rel.starts_with(['/', '\\']) || rel.split(['/', '\\']).any(|s| s == ".." || s.contains(':')) {
                        *resp.status_mut() = tauri::http::StatusCode::FORBIDDEN;
                        return;
                    }
                    let found = scene_roots.iter().map(|r| r.join(&rel)).find(|p| p.is_file());
                    match found.and_then(|p| std::fs::read(&p).ok().map(|b| (p, b))) {
                        Some((p, bytes)) => {
                            *resp.body_mut() = bytes.into();
                            *resp.status_mut() = tauri::http::StatusCode::OK;
                            resp.headers_mut().insert("content-type", mime(&p).parse().unwrap());
                        }
                        None => {
                            // Without this the asset resolver's index.html fallback
                            // would answer 200 for optional files like damage.viewer.json.
                            *resp.body_mut() = Vec::new().into();
                            *resp.status_mut() = tauri::http::StatusCode::NOT_FOUND;
                        }
                    }
                })
                .build()?;

            let Some((home, py)) = find_backend(&exe_dir) else {
                status_loop(&win, "Processing backend not installed: offline viewer only. Pick a scene below.");
                return Ok(());
            };
            let port = TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
            match spawn_backend(&home, &py, port) {
                Ok(c) => *child.lock().unwrap() = Some(c),
                Err(e) => {
                    status_loop(&win, &format!("Could not start backend ({e}): offline viewer only."));
                    return Ok(());
                }
            }
            let child = child.clone();
            std::thread::spawn(move || {
                let start = Instant::now();
                let addr: std::net::SocketAddr = ([127, 0, 0, 1], port).into();
                loop {
                    if TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok() {
                        let url = format!("http://127.0.0.1:{port}/");
                        let _ = win.navigate(url.parse().unwrap());
                        return;
                    }
                    let exited = child.lock().unwrap().as_mut().map_or(true, |c| !matches!(c.try_wait(), Ok(None)));
                    let secs = start.elapsed().as_secs();
                    let msg = if exited {
                        "Processing backend exited during startup (log: %TEMP%\\depthwizard-backend.log). Offline viewer only.".to_string()
                    } else if secs > 600 {
                        "Processing backend did not respond within 10 minutes. Offline viewer only.".to_string()
                    } else {
                        format!("Starting processing backend from {} (loading model, {secs}s)... the upload dashboard opens automatically.", home.display())
                    };
                    let _ = win.eval(&format!("typeof setStatus==='function'&&setStatus({:?})", msg));
                    if exited || secs > 600 {
                        return;
                    }
                    std::thread::sleep(Duration::from_secs(1));
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to start DepthWizard");

    app.run(move |_, event| {
        if let tauri::RunEvent::Exit = event {
            if let Some(mut c) = child_exit.lock().unwrap().take() {
                kill_tree(&mut c);
            }
        }
    });
}

/// Repeat a status message briefly so it lands after the page has loaded.
fn status_loop(win: &tauri::WebviewWindow, msg: &str) {
    let (win, js) = (win.clone(), format!("typeof setStatus==='function'&&setStatus({:?})", msg));
    std::thread::spawn(move || {
        for _ in 0..10 {
            let _ = win.eval(&js);
            std::thread::sleep(Duration::from_millis(500));
        }
    });
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() && s.is_char_boundary(i + 3) {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
