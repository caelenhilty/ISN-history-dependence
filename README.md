# miller-lab-final
Codes to generate figures for senior thesis

$cwd = Get-Location
$outputFile = Join-Path $cwd "output.txt"
$job = Start-Job -scriptblock {Set-Location -Path $using:cwd; Start-Process -FilePath "C:\Users\caele\miniconda3\envs\millerlab\python.exe" -ArgumentList "figures/figure6/data_figure6d_psycho_parallel.py" -RedirectStandardOutput $using:outputFile -NoNewWindow -Wait} 
Receive-Job $job
Get-Job
Stop-Job -State Running