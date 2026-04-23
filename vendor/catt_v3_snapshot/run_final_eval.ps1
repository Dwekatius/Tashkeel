$python = "C:\Users\Mohanad Dwekat\.unsloth\studio\unsloth_studio\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "run_final_eval.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python interpreter not found: $python"
}

if (-not (Test-Path -LiteralPath $script)) {
    throw "Eval wrapper not found: $script"
}

& $python $script @args
