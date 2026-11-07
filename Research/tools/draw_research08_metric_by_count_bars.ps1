$ErrorActionPreference = "Stop"

$Root = (Get-Location).Path
$ResultPath = Join-Path $Root "data\research08\results\research08_augmentation_saturation.csv"
$FigureDir = Join-Path $Root "data\research08\figures"

New-Item -ItemType Directory -Force -Path $FigureDir | Out-Null

$methodOrder = @(
    "Masking Diffusion",
    "Filtered Masking Diffusion",
    "ScoreFiltered Masking Diffusion",
    "Noise injection",
    "Magnitude warping",
    "Frequency domain",
    "Time warping",
    "Comprehensive"
)

$countOrder = @(50, 100, 200, 500, 750, 1000)

$metrics = @(
    @{ Key = "precision"; Label = "Precision"; File = "research08_precision_by_augmentation_count_bar.svg" },
    @{ Key = "recall"; Label = "Recall"; File = "research08_recall_by_augmentation_count_bar.svg" },
    @{ Key = "f1"; Label = "F1-score"; File = "research08_f1_by_augmentation_count_bar.svg" },
    @{ Key = "auprc"; Label = "AUPRC"; File = "research08_auprc_by_augmentation_count_bar.svg" }
)

$colors = @{
    "Masking Diffusion" = "#3569a8"
    "Filtered Masking Diffusion" = "#4f8fbc"
    "ScoreFiltered Masking Diffusion" = "#7aa6c2"
    "Noise injection" = "#d67b35"
    "Magnitude warping" = "#b85f2f"
    "Frequency domain" = "#8f9f3a"
    "Time warping" = "#6aa84f"
    "Comprehensive" = "#8e5ea2"
}

$shortLabels = @{
    "Masking Diffusion" = "MD"
    "Filtered Masking Diffusion" = "Filtered MD"
    "ScoreFiltered Masking Diffusion" = "ScoreFiltered MD"
    "Noise injection" = "Noise"
    "Magnitude warping" = "Magnitude"
    "Frequency domain" = "Frequency"
    "Time warping" = "Time"
    "Comprehensive" = "Comprehensive"
}

function Escape-XmlText([string]$Text) {
    return [System.Security.SecurityElement]::Escape($Text)
}

function Format-Score([double]$Value) {
    return $Value.ToString("0.000", [System.Globalization.CultureInfo]::InvariantCulture)
}

$rows = Import-Csv $ResultPath | Where-Object { $methodOrder -contains $_.method }
$byMethodCount = @{}
foreach ($row in $rows) {
    $key = "$($row.method)|$($row.augmentation_count)"
    $byMethodCount[$key] = $row
}

$width = 1500
$height = 760
$left = 82
$rightPad = 34
$top = 96
$plotHeight = 430
$plotWidth = $width - $left - $rightPad
$bottom = $top + $plotHeight
$groupStep = $plotWidth / $countOrder.Count
$barGap = 3
$barWidth = [Math]::Min(18, ($groupStep - 34) / $methodOrder.Count - $barGap)
$legendX = 90
$legendY = 602

foreach ($metric in $metrics) {
    $metricKey = $metric.Key
    $metricLabel = $metric.Label
    $outPath = Join-Path $FigureDir $metric.File

    $sb = [System.Text.StringBuilder]::new()
    [void]$sb.AppendLine("<svg xmlns='http://www.w3.org/2000/svg' width='$width' height='$height' viewBox='0 0 $width $height'>")
    [void]$sb.AppendLine("<style>")
    [void]$sb.AppendLine("text { font-family: Arial, sans-serif; fill: #202020; }")
    [void]$sb.AppendLine(".title { font-size: 25px; font-weight: 700; }")
    [void]$sb.AppendLine(".subtitle { font-size: 13px; fill: #555; }")
    [void]$sb.AppendLine(".axis { stroke: #333; stroke-width: 1.2; }")
    [void]$sb.AppendLine(".grid { stroke: #dddddd; stroke-width: 1; }")
    [void]$sb.AppendLine(".tick { font-size: 12px; fill: #555; }")
    [void]$sb.AppendLine(".xlabel { font-size: 13px; font-weight: 700; fill: #333; }")
    [void]$sb.AppendLine(".ylabel { font-size: 13px; fill: #333; }")
    [void]$sb.AppendLine(".legend { font-size: 13px; fill: #333; }")
    [void]$sb.AppendLine(".value { font-size: 8px; font-weight: 700; fill: #222; }")
    [void]$sb.AppendLine("</style>")
    [void]$sb.AppendLine("<rect x='0' y='0' width='$width' height='$height' fill='#ffffff'/>")
    [void]$sb.AppendLine("<text x='$($width / 2)' y='34' class='title' text-anchor='middle'>$metricLabel by Augmentation Count</text>")
    [void]$sb.AppendLine("<text x='$($width / 2)' y='58' class='subtitle' text-anchor='middle'>Research08 final test performance across augmentation methods</text>")

    [void]$sb.AppendLine("<line x1='$left' y1='$bottom' x2='$($left + $plotWidth)' y2='$bottom' class='axis'/>")
    [void]$sb.AppendLine("<line x1='$left' y1='$top' x2='$left' y2='$bottom' class='axis'/>")

    foreach ($tick in @(0, 0.25, 0.5, 0.75, 1.0)) {
        $y = $bottom - ($tick * $plotHeight)
        $tickLabel = $tick.ToString("0.00", [System.Globalization.CultureInfo]::InvariantCulture)
        [void]$sb.AppendLine("<line x1='$left' y1='$y' x2='$($left + $plotWidth)' y2='$y' class='grid'/>")
        [void]$sb.AppendLine("<text x='$($left - 13)' y='$($y + 4)' class='tick' text-anchor='end'>$tickLabel</text>")
    }

    [void]$sb.AppendLine("<text x='24' y='$($top + $plotHeight / 2)' class='ylabel' text-anchor='middle' transform='rotate(-90 24 $($top + $plotHeight / 2))'>Score</text>")
    [void]$sb.AppendLine("<text x='$($left + $plotWidth / 2)' y='$($bottom + 64)' class='ylabel' text-anchor='middle'>Augmented anomaly windows used for training</text>")

    for ($countIdx = 0; $countIdx -lt $countOrder.Count; $countIdx++) {
        $count = $countOrder[$countIdx]
        $groupLeft = $left + ($groupStep * $countIdx)
        $groupCenter = $groupLeft + ($groupStep / 2)
        [void]$sb.AppendLine("<text x='$groupCenter' y='$($bottom + 28)' class='xlabel' text-anchor='middle'>$count</text>")

        $totalBarsWidth = ($barWidth * $methodOrder.Count) + ($barGap * ($methodOrder.Count - 1))
        $firstBarX = $groupCenter - ($totalBarsWidth / 2)

        for ($methodIdx = 0; $methodIdx -lt $methodOrder.Count; $methodIdx++) {
            $method = $methodOrder[$methodIdx]
            $rowKey = "$method|$count"
            if (-not $byMethodCount.ContainsKey($rowKey)) {
                continue
            }
            $row = $byMethodCount[$rowKey]
            $value = [double]$row.$metricKey
            $barHeight = $value * $plotHeight
            $x = $firstBarX + (($barWidth + $barGap) * $methodIdx)
            $y = $bottom - $barHeight
            $color = $colors[$method]
            [void]$sb.AppendLine("<rect x='$x' y='$y' width='$barWidth' height='$barHeight' rx='1.5' fill='$color'/>")
        }
    }

    for ($i = 0; $i -lt $methodOrder.Count; $i++) {
        $method = $methodOrder[$i]
        $x = $legendX + (($i % 4) * 330)
        $y = $legendY + ([Math]::Floor($i / 4) * 32)
        [void]$sb.AppendLine("<rect x='$x' y='$($y - 13)' width='16' height='16' rx='2' fill='$($colors[$method])'/>")
        [void]$sb.AppendLine("<text x='$($x + 24)' y='$y' class='legend'>$(Escape-XmlText $shortLabels[$method])</text>")
    }

    [void]$sb.AppendLine("</svg>")
    [System.IO.File]::WriteAllText($outPath, $sb.ToString(), [System.Text.Encoding]::UTF8)
    Write-Output $outPath
}
