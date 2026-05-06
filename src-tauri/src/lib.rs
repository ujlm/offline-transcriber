use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{AppHandle, Emitter, Manager, State};

/// A long-lived Python sidecar in `--server` mode.
///
/// We spawn one Python process on the first `process_audio` call and reuse it
/// for the rest of the app's lifetime. The Cohere ASR weights stay loaded in
/// that process, so files 2..N skip the ~22 s model load. New jobs are sent
/// as a single JSON line on the child's stdin.
struct WarmSidecar {
    child: Child,
    stdin: ChildStdin,
}

#[derive(Default, Clone)]
struct SidecarState {
    sidecar: Arc<Mutex<Option<WarmSidecar>>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TranscribeArgs {
    input_path: String,
    output_dir: Option<String>,
    /// ISO 639-1 language code (e.g. "en", "fr"). Defaults to "en" if
    /// the frontend doesn't supply one. Validated by the Python sidecar
    /// against the model's `supported_languages`.
    language: Option<String>,
    /// When false, the Python sidecar skips pyannote and emits a
    /// single-speaker transcript.
    diarization_enabled: Option<bool>,
}

const EVENT_NAME: &str = "transcription-event";

#[tauri::command]
async fn process_audio(
    app: AppHandle,
    state: State<'_, SidecarState>,
    args: TranscribeArgs,
) -> Result<(), String> {
    let input_path = PathBuf::from(&args.input_path);
    let output_dir = args
        .output_dir
        .map(PathBuf::from)
        .or_else(|| input_path.parent().map(|p| p.to_path_buf()))
        .ok_or_else(|| "could not determine output directory".to_string())?;

    let language = args.language.unwrap_or_else(|| "en".to_string());
    let diarization_enabled = args.diarization_enabled.unwrap_or(true);
    let job = serde_json::json!({
        "input": input_path.to_string_lossy(),
        "output_dir": output_dir.to_string_lossy(),
        "language": language,
        "diarization_enabled": diarization_enabled,
    });
    let mut line = serde_json::to_string(&job).map_err(|e| e.to_string())?;
    line.push('\n');

    // Try to reuse a live sidecar; if it crashed or never existed, (re)spawn.
    // We retry once after a respawn so a stale child that died between checks
    // doesn't immediately surface as an error to the user.
    for attempt in 0..2 {
        ensure_sidecar_alive(&app, &state)?;
        let mut guard = state.sidecar.lock().map_err(|e| e.to_string())?;
        let sidecar = guard
            .as_mut()
            .ok_or_else(|| "sidecar unexpectedly missing".to_string())?;
        match sidecar
            .stdin
            .write_all(line.as_bytes())
            .and_then(|_| sidecar.stdin.flush())
        {
            Ok(()) => return Ok(()),
            Err(e) => {
                // Likely a broken pipe: the child died between our liveness
                // check and the write. Drop it and let the next loop iter
                // respawn. On the second failure, surface the error.
                *guard = None;
                if attempt == 1 {
                    return Err(format!("failed to send job to sidecar: {e}"));
                }
            }
        }
    }
    Err("failed to send job to sidecar after retry".into())
}

#[tauri::command]
async fn cancel_transcription(state: State<'_, SidecarState>) -> Result<(), String> {
    // Cooperative cancel isn't possible while the sidecar is mid-`generate()`
    // (it can't yield), so we kill the warm process. The next process_audio
    // call will respawn and pay the cold-start cost once. That's acceptable
    // because cancels are rare and the user already chose to abort.
    let mut guard = state.sidecar.lock().map_err(|e| e.to_string())?;
    if let Some(mut sidecar) = guard.take() {
        let _ = sidecar.child.kill();
        let _ = sidecar.child.wait();
    }
    Ok(())
}

/// Make sure `state.sidecar` holds a live, responsive child. If the slot is
/// empty or the previous child has exited, spawn a fresh one.
fn ensure_sidecar_alive(app: &AppHandle, state: &SidecarState) -> Result<(), String> {
    {
        let mut guard = state.sidecar.lock().map_err(|e| e.to_string())?;
        if let Some(sidecar) = guard.as_mut() {
            match sidecar.child.try_wait() {
                Ok(None) => return Ok(()), // alive
                Ok(Some(_)) => {
                    *guard = None;
                }
                Err(_) => {
                    *guard = None;
                }
            }
        }
    }
    spawn_sidecar(app, state)
}

fn spawn_sidecar(app: &AppHandle, state: &SidecarState) -> Result<(), String> {
    let python = resolve_python();
    let main_py = resolve_backend_main(app)?;
    let models_dir = resolve_models_dir(app)?;

    let mut cmd = Command::new(&python);
    cmd.arg("-u")
        .arg(&main_py)
        .arg("--server")
        .arg("--models-dir")
        .arg(&models_dir)
        .arg("--language")
        .arg("en")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("HF_DATASETS_OFFLINE", "1")
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to spawn python sidecar ({}): {e}", python))?;

    let stdin = child
        .stdin
        .take()
        .ok_or("no stdin on sidecar")?;
    let stdout = child.stdout.take().ok_or("no stdout on sidecar")?;
    let stderr = child.stderr.take().ok_or("no stderr on sidecar")?;

    // stdout reader: forwards JSON event lines until the child exits.
    {
        let app = app.clone();
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                forward_line(&app, &line);
            }
        });
    }

    // stderr reader: surfaces tracebacks as `log` events with level=stderr.
    {
        let app = app.clone();
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                if line.trim().is_empty() {
                    continue;
                }
                let payload = serde_json::json!({
                    "event": "log",
                    "level": "stderr",
                    "message": line,
                });
                let _ = app.emit(EVENT_NAME, payload);
            }
        });
    }

    let mut guard = state.sidecar.lock().map_err(|e| e.to_string())?;
    *guard = Some(WarmSidecar { child, stdin });
    Ok(())
}

fn forward_line(app: &AppHandle, line: &str) {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return;
    }
    let payload: Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(_) => serde_json::json!({
            "event": "log",
            "level": "info",
            "message": trimmed,
        }),
    };
    let _ = app.emit(EVENT_NAME, payload);
}

fn resolve_python() -> String {
    std::env::var("OFFLINE_TRANSCRIBER_PYTHON").unwrap_or_else(|_| {
        if cfg!(target_os = "windows") {
            "python".into()
        } else {
            "python3".into()
        }
    })
}

fn resolve_backend_main(app: &AppHandle) -> Result<PathBuf, String> {
    if let Ok(p) = std::env::var("OFFLINE_TRANSCRIBER_BACKEND_DIR") {
        let main = PathBuf::from(p).join("main.py");
        if main.exists() {
            return Ok(main);
        }
    }
    for candidate in ["backend/main.py", "../backend/main.py"] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return p
                .canonicalize()
                .map_err(|e| format!("canonicalize {p:?}: {e}"));
        }
    }
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("resource_dir: {e}"))?;
    let main = resource_dir.join("backend").join("main.py");
    if main.exists() {
        return Ok(main);
    }
    Err(format!(
        "could not locate backend/main.py (tried env, ./backend/main.py, ../backend/main.py, {})",
        main.display()
    ))
}

fn resolve_models_dir(app: &AppHandle) -> Result<PathBuf, String> {
    if let Ok(p) = std::env::var("OFFLINE_TRANSCRIBER_MODELS_DIR") {
        let dir = PathBuf::from(p);
        if dir.is_dir() {
            return Ok(dir);
        }
    }
    for candidate in ["models", "../models"] {
        let p = PathBuf::from(candidate);
        if p.is_dir() {
            return p
                .canonicalize()
                .map_err(|e| format!("canonicalize {p:?}: {e}"));
        }
    }
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("resource_dir: {e}"))?;
    let dir = resource_dir.join("models");
    if dir.is_dir() {
        return Ok(dir);
    }
    Err("could not locate models directory".into())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .manage(SidecarState::default())
        .on_window_event(|window, event| {
            // When the main window closes, kill the sidecar so we don't
            // leave an orphaned ~5 GB Python process behind.
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(state) = window.try_state::<SidecarState>() {
                    if let Ok(mut guard) = state.sidecar.lock() {
                        if let Some(mut sidecar) = guard.take() {
                            let _ = sidecar.child.kill();
                            let _ = sidecar.child.wait();
                        }
                    }
                }
            }
        })
        .invoke_handler(tauri::generate_handler![process_audio, cancel_transcription])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
