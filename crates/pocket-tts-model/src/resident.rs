//! Resident official Rust/XN engine bridge.
//!
//! Kyutai already implements fully streaming voice-cloned synthesis. Retain its
//! model and reusable URL-voice cache across requests instead of relaunching
//! the executable. The native Candle implementation will use this same host
//! interface once checkpoint parity is established.

use crate::{inspect_wav, GeneratedAudio, SynthesisEngine, SynthesisRequest, Voice};
use anyhow::{bail, Context, Result};
use reqwest::blocking::{multipart, Client};
use std::fs::File;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

/// Timings measured at the calling Rust process, including loopback HTTP.
/// First PCM excludes the initial 44-byte WAV header.
#[derive(Debug, Clone)]
pub struct StreamTiming {
    pub headers: Duration,
    pub first_pcm: Duration,
    pub total: Duration,
    pub received_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct ResidentOptions {
    pub language: String,
    pub default_voice: String,
    pub quantize: bool,
    pub threads: usize,
    pub startup_timeout: Duration,
    pub request_timeout: Duration,
}

impl Default for ResidentOptions {
    fn default() -> Self {
        Self {
            language: "english".to_owned(),
            default_voice: "alba".to_owned(),
            quantize: false,
            threads: 4,
            startup_timeout: Duration::from_secs(180),
            request_timeout: Duration::from_secs(120),
        }
    }
}

/// Owned local Rust server process. Drop kills and reaps only the child it
/// started; neither external URLs nor shell execution are accepted.
pub struct ResidentRustServer {
    child: Mutex<Child>,
    client: Client,
    url: String,
    language: String,
    binary: PathBuf,
    quantize: bool,
}

impl ResidentRustServer {
    pub fn start(binary: impl AsRef<Path>, opts: ResidentOptions) -> Result<Self> {
        let binary = binary.as_ref();
        anyhow::ensure!(
            binary.is_file(),
            "native TTS executable missing: {}",
            binary.display()
        );
        anyhow::ensure!(!opts.language.trim().is_empty(), "language cannot be empty");
        anyhow::ensure!(
            !opts.default_voice.trim().is_empty(),
            "default voice cannot be empty"
        );
        anyhow::ensure!(opts.threads > 0, "CPU threads must be positive");
        anyhow::ensure!(
            !opts.startup_timeout.is_zero(),
            "startup timeout must be positive"
        );
        anyhow::ensure!(
            !opts.request_timeout.is_zero(),
            "request timeout must be positive"
        );

        let socket = TcpListener::bind(("127.0.0.1", 0)).context("finding local TTS port")?;
        let port = socket.local_addr()?.port();
        // The official executable binds its own port. No model data is sent to
        // external hosts through this endpoint.
        drop(socket);
        let mut command = Command::new(binary);
        command.args([
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--language",
            &opts.language,
            "--default-voice",
            &opts.default_voice,
        ]);
        if opts.quantize {
            command.arg("--quantize");
        }
        let mut child = command
            .env("RAYON_NUM_THREADS", opts.threads.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("starting pinned native Rust inference server")?;
        let client = match Client::builder().timeout(opts.request_timeout).build() {
            Ok(client) => client,
            Err(err) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(err.into());
            }
        };
        let url = format!("http://127.0.0.1:{port}");
        let start = Instant::now();
        loop {
            if let Some(status) = child
                .try_wait()
                .context("reading native server exit status")?
            {
                bail!("native Rust inference server exited during startup: {status}");
            }
            if let Ok(response) = client
                .get(format!("{url}/health"))
                .timeout(Duration::from_millis(400))
                .send()
            {
                if response.status().is_success() {
                    break;
                }
            }
            if start.elapsed() >= opts.startup_timeout {
                let _ = child.kill();
                let _ = child.wait();
                bail!(
                    "native Rust server not ready after {:?}",
                    opts.startup_timeout
                );
            }
            thread::sleep(Duration::from_millis(50));
        }
        Ok(Self {
            child: Mutex::new(child),
            client,
            url,
            language: opts.language,
            binary: binary.to_path_buf(),
            quantize: opts.quantize,
        })
    }

    pub fn binary(&self) -> &Path {
        &self.binary
    }

    pub fn endpoint(&self) -> &str {
        &self.url
    }

    /// Deliver HTTP-streamed WAV bytes through a caller-controlled callback,
    /// without buffering entire synthesized speech in this library.
    pub fn stream_wav(
        &self,
        request: &SynthesisRequest,
        mut on_bytes: impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<StreamTiming> {
        request.validate()?;
        anyhow::ensure!(
            request.language == self.language,
            "the resident model is loaded for {:?}, request asked for {:?}",
            self.language,
            request.language
        );
        anyhow::ensure!(
            request.quantize == self.is_quantized(),
            "quantization is chosen when starting the resident model"
        );
        let post = self.client.post(format!("{}/tts", self.url));
        let post = match &request.voice {
            Voice::Preset(name) => post.form(&[
                ("text", request.text.as_str()),
                ("voice_url", name.as_str()),
            ]),
            Voice::HubAudio(url) => {
                post.form(&[("text", request.text.as_str()), ("voice_url", url.as_str())])
            }
            Voice::ReferenceAudio(path) | Voice::State(path) => {
                let filename = path
                    .file_name()
                    .and_then(|v| v.to_str())
                    .context("local voice file does not have UTF-8 filename")?;
                let data = std::fs::read(path)
                    .with_context(|| format!("reading voice input {}", path.display()))?;
                let part = multipart::Part::bytes(data)
                    .file_name(filename.to_string())
                    .mime_str("application/octet-stream")?;
                post.multipart(
                    multipart::Form::new()
                        .text("text", request.text.clone())
                        .part("voice_wav", part),
                )
            }
        };
        let start = Instant::now();
        let mut response = post.send().context("requesting resident voice synthesis")?;
        let headers = start.elapsed();
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().unwrap_or_default();
            bail!(
                "resident voice synthesis failed: {status}: {}",
                text.chars().take(300).collect::<String>()
            );
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        anyhow::ensure!(
            content_type.contains("audio/wav"),
            "unexpected audio response content-type"
        );
        let mut buf = [0_u8; 64 * 1024];
        let mut prefix = Vec::with_capacity(44);
        let mut total_bytes = 0_u64;
        let mut first_pcm = None;
        loop {
            let count = response
                .read(&mut buf)
                .context("reading streaming WAV response")?;
            if count == 0 {
                break;
            }
            if prefix.len() < 44 {
                let copy = (44 - prefix.len()).min(count);
                prefix.extend_from_slice(&buf[..copy]);
            }
            total_bytes += count as u64;
            if first_pcm.is_none() && total_bytes > 44 {
                first_pcm = Some(start.elapsed());
            }
            on_bytes(&buf[..count])?;
        }
        anyhow::ensure!(
            prefix.len() == 44
                && &prefix[0..4] == b"RIFF"
                && &prefix[8..12] == b"WAVE"
                && &prefix[36..40] == b"data",
            "resident inference did not return valid streaming PCM WAV"
        );
        anyhow::ensure!(
            first_pcm.is_some(),
            "resident TTS returned WAV header without audio"
        );
        Ok(StreamTiming {
            headers,
            first_pcm: first_pcm.context("missing first PCM")?,
            total: start.elapsed(),
            received_bytes: total_bytes,
        })
    }

    fn is_quantized(&self) -> bool {
        self.quantize
    }
}

impl SynthesisEngine for ResidentRustServer {
    fn synthesize_wav(&self, request: &SynthesisRequest, output: &Path) -> Result<GeneratedAudio> {
        anyhow::ensure!(
            output.extension().and_then(|v| v.to_str()) == Some("wav"),
            "output must have .wav extension"
        );
        let parent = output.parent().unwrap_or_else(|| Path::new("."));
        anyhow::ensure!(
            parent.as_os_str().is_empty() || parent.is_dir(),
            "output directory missing: {}",
            parent.display()
        );
        let mut writer = File::create(output)
            .with_context(|| format!("creating synthesized WAV {}", output.display()))?;
        let result = self.stream_wav(request, |chunk| {
            writer.write_all(chunk).context("writing generated audio")
        });
        if let Err(err) = result {
            drop(writer);
            let _ = std::fs::remove_file(output);
            return Err(err);
        }
        writer.flush()?;
        drop(writer);
        match inspect_wav(output) {
            Ok(audio) => Ok(audio),
            Err(err) => {
                let _ = std::fs::remove_file(output);
                Err(err)
            }
        }
    }
}

impl Drop for ResidentRustServer {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn options_reject_missing_executable() {
        let opts = ResidentOptions::default();
        assert!(ResidentRustServer::start("/path/absent/pocket-tts", opts).is_err());
    }
}
