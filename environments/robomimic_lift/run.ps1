param(
    [ValidateSet('setup', 'validate')]
    [string]$Stage = 'validate'
)
$liftProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$liftWindowsPath = $liftProjectRoot.Replace('\', '/')
$liftPathOutput = & wsl.exe -d Ubuntu -- wslpath -u $liftWindowsPath
$liftPathExit = $LASTEXITCODE
if ($liftPathExit -ne 0) { throw "Cannot resolve project path in WSL Ubuntu (exit $liftPathExit, path $liftWindowsPath): $liftPathOutput" }
$liftLinuxRoot = ($liftPathOutput | Out-String).Trim()
$liftScript = if ($Stage -eq 'setup') { 'setup_wsl.sh' } else { 'validate_wsl.sh' }
& wsl.exe -d Ubuntu -- bash "$liftLinuxRoot/environments/robomimic_lift/$liftScript"
if ($LASTEXITCODE -ne 0) { throw "Lift $Stage failed (exit $LASTEXITCODE)" }
