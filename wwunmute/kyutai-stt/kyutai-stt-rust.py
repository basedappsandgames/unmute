
import subprocess
from pathlib import Path

import modal

this_directory = Path(__file__).parent.resolve()

app = modal.App(name="kyutai-stt-rust")

stt_image = (
    modal.Image.from_registry(
    f"nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.10")
    .entrypoint([])
    .apt_install([
        "build-essential",
        "pkg-config",
        "curl",
        "cmake",
        "clang",
        "openssl",
        "libssl-dev",
        "git",
        "python3-dev",
    ])
    .env({
        "PATH": "/root/.cargo/bin:${PATH}",
    })    
    .run_commands([
        "curl https://sh.rustup.rs -sSf | bash -s -- -y",
        "cargo install --features cuda moshi-server",
    ], gpu="L40S")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(this_directory / "kyutai-stt-rust.toml", "/root/kyutai-stt-rust.toml")
)

MODEL_NAME = "kyutai/stt-1b-en_fr"

hf_cache_vol = modal.Volume.from_name(f"{app.name}-hf-cache", create_if_missing=True)
hf_cache_vol_path = Path("/root/.cache/huggingface")

logs_vol = modal.Volume.from_name(f"{app.name}-logs", create_if_missing=True)
logs_vol_path = Path("/root/tmp/tts-logs")
volumes = {
    hf_cache_vol_path: hf_cache_vol,
    logs_vol_path: logs_vol
}

MINUTES = 60

@app.function(
    image=stt_image, 
    gpu="L4", 
    volumes=volumes, 
    timeout=10 * MINUTES,
    min_containers=1,
    buffer_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(
    port=8080,
    startup_timeout = 20 * MINUTES
)
def kyutai_stt_server():

    subprocess.Popen(
        [
            'moshi-server worker --config /root/kyutai-stt-rust.toml --addr 0.0.0.0 --port 8080 --log debug'
        ],
        shell=True
    )