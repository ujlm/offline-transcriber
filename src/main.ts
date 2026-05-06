import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open } from "@tauri-apps/plugin-dialog";
import { revealItemInDir } from "@tauri-apps/plugin-opener";

type TranscriptTurn = {
  start: number;
  end: number;
  speaker: string;
  text: string;
};

type SidecarEvent =
  | { event: "progress"; stage: string; percent: number }
  | { event: "log"; level: string; message: string }
  | {
      event: "done";
      txt_path: string;
      docx_path: string;
      profile_path?: string;
      transcript?: TranscriptTurn[];
      diarization_enabled?: boolean;
    }
  | { event: "cancelled" }
  | { event: "error"; code: string; message: string };

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} not found`);
  return el as T;
};

const els = {
  pickInput: $<HTMLButtonElement>("pick-input"),
  pickOutput: $<HTMLButtonElement>("pick-output"),
  dropZone: $<HTMLDivElement>("drop-zone"),
  inputPath: $<HTMLSpanElement>("input-path"),
  outputPath: $<HTMLSpanElement>("output-path"),
  language: $<HTMLSelectElement>("language"),
  diarization: $<HTMLInputElement>("diarization"),
  start: $<HTMLButtonElement>("start"),
  cancel: $<HTMLButtonElement>("cancel"),
  progress: $<HTMLDivElement>("progress"),
  barFill: $<HTMLDivElement>("bar-fill"),
  stageLabel: $<HTMLSpanElement>("stage-label"),
  percentLabel: $<HTMLSpanElement>("percent-label"),
  result: $<HTMLDivElement>("result"),
  resultText: $<HTMLParagraphElement>("result-text"),
  openOutput: $<HTMLButtonElement>("open-output"),
  error: $<HTMLDivElement>("error"),
  timecodeView: $<HTMLElement>("timecode-view"),
  audio: $<HTMLAudioElement>("audio"),
  playPause: $<HTMLButtonElement>("play-pause"),
  currentTime: $<HTMLSpanElement>("current-time"),
  duration: $<HTMLSpanElement>("duration"),
  seek: $<HTMLInputElement>("seek"),
  transcriptList: $<HTMLDivElement>("transcript-list"),
};

const state: {
  inputPath: string | null;
  outputDir: string | null;
  lastDocxPath: string | null;
  unlisten: UnlistenFn | null;
  transcript: TranscriptTurn[];
  busy: boolean;
} = {
  inputPath: null,
  outputDir: null,
  lastDocxPath: null,
  unlisten: null,
  transcript: [],
  busy: false,
};

const STAGE_LABELS: Record<string, string> = {
  decode: "Decoding audio…",
  vad: "Detecting speech…",
  diarize: "Handling speaker labels…",
  chunk: "Preparing transcript chunks…",
  transcribe: "Transcribing…",
  fusion: "Aligning speakers and text…",
  export: "Writing output files…",
};

const AUDIO_EXTENSIONS = new Set(["wav", "flac", "ogg", "mp3", "m4a"]);

function setError(msg: string | null) {
  if (msg) {
    els.error.textContent = msg;
    els.error.hidden = false;
  } else {
    els.error.hidden = true;
    els.error.textContent = "";
  }
}

function refreshStartButton() {
  els.start.disabled = state.busy || state.inputPath === null;
}

function isAudioPath(path: string) {
  const ext = path.split(".").pop()?.toLowerCase();
  return ext !== undefined && AUDIO_EXTENSIONS.has(ext);
}

function setInputPath(path: string) {
  state.inputPath = path;
  state.transcript = [];
  els.inputPath.textContent = path;
  els.audio.src = convertFileSrc(path);
  els.audio.load();
  els.result.hidden = true;
  els.timecodeView.hidden = true;
  els.transcriptList.innerHTML = "";
  refreshStartButton();
}

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds) || seconds < 0) return "00:00";
  const total = Math.floor(seconds);
  const hrs = Math.floor(total / 3600);
  const mins = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hrs > 0) {
    return `${hrs}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function renderTranscript(turns: TranscriptTurn[], diarizationEnabled: boolean) {
  state.transcript = turns;
  els.transcriptList.innerHTML = "";

  if (turns.length === 0) {
    els.transcriptList.textContent = "No transcript segments were produced.";
    els.timecodeView.hidden = false;
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const turn of turns) {
    const item = document.createElement("article");
    item.className = "transcript-item";
    item.dataset.start = String(turn.start);
    item.dataset.end = String(turn.end);

    const timeButton = document.createElement("button");
    timeButton.type = "button";
    timeButton.className = "timecode";
    timeButton.textContent = formatTime(turn.start);
    timeButton.addEventListener("click", () => {
      els.audio.currentTime = turn.start;
      void els.audio.play();
    });

    const speaker = document.createElement("div");
    speaker.className = "speaker";
    speaker.textContent = diarizationEnabled ? turn.speaker : "Transcript";

    const text = document.createElement("p");
    text.className = "utterance";
    text.textContent = turn.text;

    item.append(timeButton, speaker, text);
    fragment.append(item);
  }

  els.transcriptList.append(fragment);
  els.timecodeView.hidden = false;
}

function updateActiveTurn() {
  const time = els.audio.currentTime;
  for (const item of els.transcriptList.querySelectorAll<HTMLElement>(".transcript-item")) {
    const start = Number(item.dataset.start);
    const end = Number(item.dataset.end);
    item.classList.toggle("active", Number.isFinite(start) && time >= start && time < end);
  }
}

function updatePlayer() {
  const current = els.audio.currentTime || 0;
  const duration = Number.isFinite(els.audio.duration) ? els.audio.duration : 0;
  els.currentTime.textContent = formatTime(current);
  els.duration.textContent = formatTime(duration);
  els.seek.max = String(duration);
  els.seek.value = String(Math.min(current, duration || current));
  updateActiveTurn();
}

function setBusy(busy: boolean) {
  state.busy = busy;
  els.pickInput.disabled = busy;
  els.pickOutput.disabled = busy;
  els.language.disabled = busy;
  els.diarization.disabled = busy;
  els.cancel.hidden = !busy;
  refreshStartButton();
}

els.pickInput.addEventListener("click", async () => {
  setError(null);
  const picked = await open({
    multiple: false,
    directory: false,
    filters: [{ name: "Audio", extensions: ["wav", "flac", "ogg", "mp3", "m4a"] }],
  });
  if (typeof picked === "string") {
    setInputPath(picked);
  }
});

els.dropZone.addEventListener("click", (event) => {
  if (state.busy) return;
  if (event.target instanceof HTMLButtonElement) return;
  els.pickInput.click();
});

els.dropZone.addEventListener("keydown", (event) => {
  if (state.busy) return;
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    els.pickInput.click();
  }
});

void getCurrentWebview().onDragDropEvent((event) => {
  const payload = event.payload;
  if (state.busy) {
    els.dropZone.classList.remove("drag-over");
    return;
  }
  if (payload.type === "enter" || payload.type === "over") {
    els.dropZone.classList.add("drag-over");
    return;
  }
  els.dropZone.classList.remove("drag-over");
  if (payload.type !== "drop") return;

  const [path] = payload.paths;
  if (!path) return;
  if (!isAudioPath(path)) {
    setError("Please drop a supported audio file: wav, flac, ogg, mp3, or m4a.");
    return;
  }
  setError(null);
  setInputPath(path);
});

els.pickOutput.addEventListener("click", async () => {
  setError(null);
  const picked = await open({ multiple: false, directory: true });
  if (typeof picked === "string") {
    state.outputDir = picked;
    els.outputPath.textContent = picked;
  }
});

els.start.addEventListener("click", async () => {
  if (!state.inputPath) return;
  setError(null);
  state.lastDocxPath = null;
  state.transcript = [];

  setBusy(true);
  els.result.hidden = true;
  els.timecodeView.hidden = true;
  els.progress.hidden = false;
  els.barFill.style.width = "0%";
  els.stageLabel.textContent = "Starting…";
  els.percentLabel.textContent = "0%";

  state.unlisten = await listen<SidecarEvent>("transcription-event", (e) => {
    const payload = e.payload;
    switch (payload.event) {
      case "progress": {
        els.barFill.style.width = `${payload.percent}%`;
        els.percentLabel.textContent = `${Math.round(payload.percent)}%`;
        els.stageLabel.textContent = STAGE_LABELS[payload.stage] ?? payload.stage;
        break;
      }
      case "log": {
        console.log(`[sidecar:${payload.level}]`, payload.message);
        break;
      }
      case "done": {
        state.lastDocxPath = payload.docx_path;
        els.barFill.style.width = "100%";
        els.percentLabel.textContent = "100%";
        els.stageLabel.textContent = "Done";
        els.result.hidden = false;
        els.resultText.textContent = "Transcription complete.";
        renderTranscript(payload.transcript ?? [], payload.diarization_enabled ?? true);
        if (payload.profile_path) {
          console.log(`[sidecar] profile written to ${payload.profile_path}`);
        }
        finalize();
        break;
      }
      case "cancelled": {
        els.stageLabel.textContent = "Cancelled";
        finalize();
        break;
      }
      case "error": {
        setError(`${payload.code}: ${payload.message}`);
        finalize();
        break;
      }
    }
  });

  try {
    await invoke("process_audio", {
      args: {
        input_path: state.inputPath,
        output_dir: state.outputDir,
        language: els.language.value || "en",
        diarization_enabled: els.diarization.checked,
      },
    });
  } catch (err) {
    setError(String(err));
    finalize();
  }
});

els.cancel.addEventListener("click", async () => {
  try {
    await invoke("cancel_transcription");
  } catch (err) {
    setError(String(err));
  }
});

els.openOutput.addEventListener("click", async () => {
  if (state.lastDocxPath) {
    await revealItemInDir(state.lastDocxPath);
  }
});

els.playPause.addEventListener("click", async () => {
  if (els.audio.paused) {
    try {
      await els.audio.play();
    } catch (err) {
      setError(String(err));
    }
  } else {
    els.audio.pause();
  }
});

els.seek.addEventListener("input", () => {
  els.audio.currentTime = Number(els.seek.value);
});

els.audio.addEventListener("loadedmetadata", updatePlayer);
els.audio.addEventListener("timeupdate", updatePlayer);
els.audio.addEventListener("play", () => {
  els.playPause.textContent = "Pause";
  els.playPause.setAttribute("aria-label", "Pause audio");
});
els.audio.addEventListener("pause", () => {
  els.playPause.textContent = "Play";
  els.playPause.setAttribute("aria-label", "Play audio");
});
els.audio.addEventListener("ended", () => {
  els.playPause.textContent = "Play";
  els.playPause.setAttribute("aria-label", "Play audio");
});

function finalize() {
  setBusy(false);
  if (state.unlisten) {
    state.unlisten();
    state.unlisten = null;
  }
}
