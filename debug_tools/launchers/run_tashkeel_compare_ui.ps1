$python = "C:\Users\Mohanad Dwekat\.unsloth\studio\unsloth_studio\Scripts\python.exe"
$script = Join-Path (Split-Path -Parent $PSScriptRoot) "tashkeel_compare_ui.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python interpreter not found: $python"
}

if (-not (Test-Path -LiteralPath $script)) {
    throw "Compare UI script not found: $script"
}

& $python $script @args
