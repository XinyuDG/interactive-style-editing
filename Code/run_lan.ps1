# conda activate torch118
$env:CLOTH_SERVER_NAME = "0.0.0.0"
$env:CLOTH_SERVER_PORT = "7860"
Write-Host "Use ipconfig to find your VPN/LAN IPv4 address."
Write-Host "Teammates can open http://<your-ip>:7860"
Write-Host "If not accessible, check Windows Firewall."
python nn\gradioFinal.py
