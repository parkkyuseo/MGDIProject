[CmdletBinding()]
param(
    [string]$BindAddress = "0.0.0.0",
    [int]$Port = 19620,
    [string]$OutputDir = "",
    [switch]$Once
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Get-SafeFilePart {
    param([string]$Text, [string]$Fallback = "UNKNOWN")

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return $Fallback
    }

    $safe = ($Text -replace '[^A-Za-z0-9._-]', '_').Trim('_')
    if ([string]::IsNullOrWhiteSpace($safe)) {
        return $Fallback
    }

    return $safe
}

function Get-SessionCsvPath {
    param(
        [string]$BaseDir,
        [pscustomobject]$Envelope
    )

    $participant = Get-SafeFilePart -Text $Envelope.participant_id -Fallback "TEST"
    $session = Get-SafeFilePart -Text $Envelope.session_timestamp -Fallback "SESSION"
    $name = "StudyMirror_{0}_{1}.csv" -f $participant, $session
    return Join-Path $BaseDir $name
}

function Append-JsonLine {
    param(
        [string]$Path,
        [string]$Line
    )

    Add-Content -LiteralPath $Path -Value $Line -Encoding UTF8
}

function Ensure-CsvRowWritten {
    param(
        [string]$CsvPath,
        [pscustomobject]$Envelope
    )

    $header = [string]$Envelope.csv_header
    $row = [string]$Envelope.csv_row

    if ([string]::IsNullOrWhiteSpace($row)) {
        throw "Received envelope without csv_row."
    }

    $needHeader = -not (Test-Path -LiteralPath $CsvPath)
    if ($needHeader) {
        if ([string]::IsNullOrWhiteSpace($header)) {
            throw "Received first row for '$CsvPath' without csv_header."
        }

        Set-Content -LiteralPath $CsvPath -Value $header -Encoding UTF8
    }

    Add-Content -LiteralPath $CsvPath -Value $row -Encoding UTF8
}

function Read-LineFromStream {
    param([System.Net.Sockets.NetworkStream]$Stream)

    $reader = New-Object System.IO.StreamReader(
        $Stream,
        [System.Text.Encoding]::UTF8,
        $false,
        4096,
        $true
    )

    return $reader.ReadLine()
}

function Write-Ack {
    param(
        [System.Net.Sockets.NetworkStream]$Stream,
        [string]$RowId,
        [bool]$Ok,
        [string]$Status,
        [string]$ErrorText = ""
    )

    $ack = [ordered]@{
        type   = "ack"
        row_id = $RowId
        ok     = $Ok
        status = $Status
        error  = $ErrorText
    } | ConvertTo-Json -Compress

    $writer = New-Object System.IO.StreamWriter(
        $Stream,
        [System.Text.Encoding]::UTF8,
        4096,
        $true
    )
    $writer.NewLine = "`n"
    $writer.WriteLine($ack)
    $writer.Flush()
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $PSScriptRoot "StudyLoggerMirrorInbox"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$jsonlPath = Join-Path $OutputDir "received_rows.jsonl"

$ipAddress = if ($BindAddress -eq "0.0.0.0") {
    [System.Net.IPAddress]::Any
} else {
    [System.Net.IPAddress]::Parse($BindAddress)
}

$listener = [System.Net.Sockets.TcpListener]::new($ipAddress, $Port)
$listener.Server.SetSocketOption(
    [System.Net.Sockets.SocketOptionLevel]::Socket,
    [System.Net.Sockets.SocketOptionName]::ReuseAddress,
    $true
)
$listener.Start()

Write-Host ("[MirrorReceiver] Listening on {0}:{1}" -f $BindAddress, $Port)
Write-Host ("[MirrorReceiver] Output directory: {0}" -f $OutputDir)

$receivedCount = 0

try {
    while ($true) {
        $client = $listener.AcceptTcpClient()
        $client.ReceiveTimeout = 5000
        $client.SendTimeout = 5000

        try {
            $stream = $client.GetStream()
            $line = Read-LineFromStream -Stream $stream

            if ([string]::IsNullOrWhiteSpace($line)) {
                Write-Ack -Stream $stream -RowId "" -Ok $false -Status "error" -ErrorText "empty_line"
                continue
            }

            $envelope = $line | ConvertFrom-Json
            if ($null -eq $envelope) {
                throw "Failed to parse JSON."
            }

            if ($envelope.type -ne "trial_row") {
                throw ("Unexpected type '{0}'." -f $envelope.type)
            }

            if ($envelope.protocol -ne "study_logger_mirror_v1") {
                throw ("Unexpected protocol '{0}'." -f $envelope.protocol)
            }

            $csvPath = Get-SessionCsvPath -BaseDir $OutputDir -Envelope $envelope
            Ensure-CsvRowWritten -CsvPath $csvPath -Envelope $envelope
            Append-JsonLine -Path $jsonlPath -Line $line

            Write-Ack -Stream $stream -RowId ([string]$envelope.row_id) -Ok $true -Status "ok"

            $receivedCount++
            Write-Host ("[MirrorReceiver] Saved row {0} for participant {1} -> {2}" -f $envelope.row_id, $envelope.participant_id, $csvPath)

            if ($Once) {
                break
            }
        }
        catch {
            try {
                if ($client.Connected) {
                    $errRowId = ""
                    try {
                        if ($null -ne $envelope -and $null -ne $envelope.row_id) {
                            $errRowId = [string]$envelope.row_id
                        }
                    }
                    catch { }

                    Write-Ack -Stream $stream -RowId $errRowId -Ok $false -Status "error" -ErrorText $_.Exception.Message
                }
            }
            catch { }

            Write-Warning ("[MirrorReceiver] {0}" -f $_.Exception.Message)
        }
        finally {
            $client.Close()
        }
    }
}
finally {
    $listener.Stop()
    Write-Host ("[MirrorReceiver] Stopped. Received {0} row(s)." -f $receivedCount)
}
