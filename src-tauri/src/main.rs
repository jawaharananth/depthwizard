// DepthWizard desktop shell.
//
// Deliberately minimal: the viewer is already a self-contained WebGPU page, so
// this exists only to give it a window and serve it from the local filesystem.
// Any logic added here would be logic the browser build no longer shares.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("failed to start DepthWizard");
}
