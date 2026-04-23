$python = "C:\Users\Mohanad Dwekat\.unsloth\studio\unsloth_studio\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "tashkeel_ui.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python interpreter not found: $python"
}

if (-not (Test-Path -LiteralPath $script)) {
    throw "UI script not found: $script"
}

& $python $script --model eo @args
