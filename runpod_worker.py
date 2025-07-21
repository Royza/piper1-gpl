def handler(event):
    import subprocess
    import os

    voice = event["voice_name"]
    csv_path = event["csv_path"]
    audio_dir = event["audio_dir"]
    espeak_voice = event["espeak_voice"]
    cache_dir = event["cache_dir"]
    config_path = event["config_path"]
    batch_size = str(event["batch_size"])
    sample_rate = str(event.get("sample_rate", 22050))
    ckpt = event.get("ckpt_path", None)

    cmd = [
        "python3", "-m", "piper.train", "fit",
        "--data.voice_name", voice,
        "--data.csv_path", csv_path,
        "--data.audio_dir", audio_dir,
        "--model.sample_rate", sample_rate,
        "--data.espeak_voice", espeak_voice,
        "--data.cache_dir", cache_dir,
        "--data.config_path", config_path,
        "--data.batch_size", batch_size
    ]

    if ckpt:
        cmd += ["--ckpt_path", ckpt]

    subprocess.run(cmd, check=True)

    # Export the ONNX model
    subprocess.run([
        "python3", "-m", "piper.train.export_onnx",
        "--checkpoint", "lightning_logs/version_0/checkpoints/last.ckpt",
        "--output-file", f"{voice}.onnx"
    ])

    return {
        "status": "complete",
        "onnx_file": f"{voice}.onnx"
    }