# Brief introduction
This project is an interactive image style transfer and local appearance editing system based on *Gradio*.
It mainly includes *three functions*:
- VGG-based Neural Style Transfer
- Stable Diffusion Img2Img refinement
- Person-aware Local Edit。


# Operation process
### 1. Environmental preparation
*Recommended:*
1. Create or activate your Python environment.
2. Use a GPU-enabled PyTorch install if you want Stable Diffusion performance.
3. Run all commands from the project root.
Example:
```powershell
conda create -n cloth python= 3.11.14
conda activate cloth
```

*Install dependencies:  *
```powershell  
pip install -r requirements.txt
```


### 2.Model preparation
Before the first run, download the required pretrained models:
*Option A: Use the setup script*
```powershell
.\setup_models.ps1
```
This script runs:
```powershell
python scripts\download_models.py
```

*Option B: Run the Python command manually*
```powershell  
python scripts\download_models.py
```
This step downloads or prepares the required pretrained models, including:
- `fashn-ai/fashn-human-parser`
- `stable-diffusion-v1-5/stable-diffusion-v1-5`
- `torchvision` VGG19 weights
The handoff package does not include model weights or cache files, so this step is required before the first run.


### 3.Local startup
*Option A: Use the local startup script*
```powershell  
.\run_local.ps1
```
This starts the app at:
```powershell  
http://127.0.0.1:7860
```

*Option B: Run the Python command manually*
```powershell  
python nn\gradioFinal.py
```
After startup, open the local URL shown in the terminal.


### 4. LAN / VPN Start
*If need to access the app through the same LAN or VPN, use:*
```powershell 
.\run_lan.ps1
```

*Or run manually:*
```powershell 
$env:CLOTH_SERVER_NAME="0.0.0.0"
$env:CLOTH_SERVER_PORT="7860"

python nn\gradioFinal.py
```
Then run:
```powershell 
ipconfig
```
Find your LAN/VPN IPv4 address and share:
```powershell 
http://<your-ip>:7860
```

If others cannot open the link, check:
- whether they are on the same LAN or VPN;
- whether the IP address is correct;
- whether Windows Firewall allows port `7860`.

### 5. Use the existing local model cache
If the required models have already been downloaded, you can force the app to use local cache only:
```powershell 
$env:HF_HOME = (Get-Location).Path
$env:HF_HUB_CACHE = Join-Path (Get-Location).Path "hub"
$env:TORCH_HOME = Join-Path (Get-Location).Path "torch_cache"

$env:CLOTH_LOCAL_FILES_ONLY = "true"
$env:CLOTH_AUTO_DOWNLOAD_MODELS = "false"

python nn\gradioFinal.py
```
This prevents the app from automatically downloading models during runtime.


# File Description

## Root-level files and folders

| File / Folder      | Description                                                                                                                                     |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `README.md`        | Main project guide. It explains the project purpose, setup steps, running commands, and file structure.                                         |
| `requirements.txt` | Python dependency list. Use `pip install -r requirements.txt` to install the required packages.                                                 |
| `.env.example`     | Example environment variables for cache paths, local/offline mode, server settings, and authentication.                                         |
| `.gitignore`       | Git ignore rules. It prevents model weights, cache files, generated results, and debug files from being committed.                              |
| `run_local.ps1`    | PowerShell script for local startup.                                                                                                            |
| `run_lan.ps1`      | PowerShell script for LAN/VPN startup.                                                                                                          |
| `setup_models.ps1` | PowerShell script for model preparation.                                                                                                        |
| `nn/`              | Core source code folder.                                                                                                                        |
| `scripts/`         | Utility scripts for model downloading and package generation.                                                                                   |

## Folder-level files
### `nn/`

| File | Description |
| --- | --- |
| `nn/gradioFinal.py` | Main Gradio application entry. It builds the UI, defines the three tabs, binds user events, and routes each workflow. |
| `nn/config.py` | Global configuration file. It manages project paths, model IDs, Hugging Face cache paths, local/offline loading, and result directory settings. |
| `nn/model.py` | VGG19-based Neural Style Transfer module. It performs feature-based style transfer from content and style images. |
| `nn/diffusion_model.py` | Stable Diffusion module. It loads and runs Img2Img refinement / generation and retains optional diffusion-related branches. |
| `nn/garment_parser.py` | Human parsing and mask generation module. It generates semantic masks for person-aware local editing. |


### `scripts/`

| File | Description |
| --- | --- |
| `scripts/download_models.py` | Downloads or prepares required pretrained models, including Human Parser, Stable Diffusion, and TorchVision VGG19 weights. |
| `scripts/package_handoff.py` | Generates a clean handoff zip while excluding model weights, caches, generated results, debug files, and checkpoints. |
