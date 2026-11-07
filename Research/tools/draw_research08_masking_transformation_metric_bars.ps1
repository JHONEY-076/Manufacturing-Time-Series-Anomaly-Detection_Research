$ErrorActionPreference = "Stop"

$Root = (Get-Location).Path
$ResultPath = Join-Path $Root "data\research08\results\research08_generative_vs_traditional.csv"
$FigureDir = Join-Path $Root "data\research08\figures"
$OutPath = Join-Path $FigureDir "research08_masking_transformation_metric_bars.svg"

New-Item -ItemType Directory -Force -Path $FigureDir | Out-Null

$methodOrder = @(
    "Masking Diffusion",
    "Masking GT-GAN",
    "Noise injection",
    "Magnitude warping",
    "Frequency domain",
    "Time warping",
    "Comprehensive"
)

$methodLabels = @{
    "Masking Diffusion" = @("Masking", "Diffusion")
    "Masking GT-GAN" = @("Masking", "GT-GAN")
    "Noise injection" = @("Noise", "injection")
    "Magnitude warping" = @("Magnitude", "warping")
    "Frequency domain" = @("Frequency", "domain")
    "Time warping" = @("Time", "warping")
    "Comprehensive" = @("Comprehensive")
}

$metrics = @(
    @{ Key = "precision"; Label = "Precision" },
    @{ Key = "recall"; Label = "Recall" },
    @{ Key = "f1"; Label = "F1-score" },
    @{ Key = "auprc"; Label = "AUPRC" }
)

$rowsByMethod = @{}
Import-Csv $ResultPath | Where-Object { $methodOrder -contains $_.method } | ForEach-Object {
    $rowsByMethod[$_.method] = $_
}

$width = 1280
$height = 820
$marginLeft = 70
$plotWidth = 540
$plotHeight = 245
$gapX = 80
$gapY = 105
$top1 = 95
$top2 = $top1 + $plotHeight + $gapY
$left1 = 70
$left2 = $left1 + $plotWidth + $gapX
$barWidth = 48
$groupStep = $plotWidth / $methodOrder.Count

function Escape-XmlText([string]$Text) {
    return [System.Security.SecurityElement]::Escape($Text)
}

function Format-Score([double]$Value) {
    return $Value.ToString("0.000", [System.Globalization.CultureInfo]::InvariantCulture)
}

function Add-AxisAndBars($Sb, [hashtable]$Metric, [int]$Left, [int]$Top) {
    $key = $Metric.Key
    $label = $Metric.Label
    $bottom = $Top + $script:plotHeight
    $right = $Left + $script:plotWidth

    [void]$Sb.AppendLine("<text x='$($Left + $script:plotWidth / 2)' y='$($Top - 22)' class='panel-title' text-anchor='middle'>$(Escape-XmlText $label)</text>")
    [void]$Sb.AppendLine("<line x1='$Left' y1='$bottom' x2='$right' y2='$bottom' class='axis'/>")
    [void]$Sb.AppendLine("<line x1='$Left' y1='$Top' x2='$Left' y2='$bottom' class='axis'/>")

    foreach ($tick in @(0, 0.25, 0.5, 0.75, 1.0)) {
        $y = $bottom - ($tick * $script:plotHeight)
        $tickLabel = $tick.ToString("0.00", [System.Globalization.CultureInfo]::InvariantCulture)
        [void]$Sb.AppendLine("<line x1='$Left' y1='$y' x2='$right' y2='$y' class='grid'/>")
        [void]$Sb.AppendLine("<text x='$($Left - 12)' y='$($y + 4)' class='tick' text-anchor='end'>$tickLabel</text>")
    }

    for ($i = 0; $i -lt $script:methodOrder.Count; $i++) {
        $method = $script:methodOrder[$i]
        $row = $script:rowsByMethod[$method]
        $value = [double]$row.$key
        $barHeight = $value * $script:plotHeight
        $xCenter = $Left + ($script:groupStep * ($i + 0.5))
        $x = $xCenter - ($script:barWidth / 2)
        $y = $bottom - $barHeight
        $category = if ($row.augmentation_family -eq "generative") { "masking" } else { "transform" }
        [void]$Sb.AppendLine("<rect x='$x' y='$y' width='$script:barWidth' height='$barHeight' rx='2' class='bar $category'/>")
        [void]$Sb.AppendLine("<text x='$xCenter' y='$($y - 7)' class='value' text-anchor='middle'>$(Format-Score $value)</text>")

        $labelY = $bottom + 22
        foreach ($part in $script:methodLabels[$method]) {
            [void]$Sb.AppendLine("<text x='$xCenter' y='$labelY' class='xlabel' text-anchor='middle'>$(Escape-XmlText $part)</text>")
            $labelY += 15
        }
    }
}

$sb = [System.Text.StringBuilder]::new()
[void]$sb.AppendLine("<svg xmlns='http://www.w3.org/2000/svg' width='$width' height='$height' viewBox='0 0 $width $height'>")
[void]$sb.AppendLine("<style>")
[void]$sb.AppendLine("text { font-family: Arial, sans-serif; fill: #202020; }")
[void]$sb.AppendLine(".title { font-size: 24px; font-weight: 700; }")
[void]$sb.AppendLine(".subtitle { font-size: 13px; fill: #555; }")
[void]$sb.AppendLine(".panel-title { font-size: 17px; font-weight: 700; }")
[void]$sb.AppendLine(".axis { stroke: #333; stroke-width: 1.2; }")
[void]$sb.AppendLine(".grid { stroke: #dddddd; stroke-width: 1; }")
[void]$sb.AppendLine(".tick { font-size: 11px; fill: #555; }")
[void]$sb.AppendLine(".xlabel { font-size: 11px; fill: #333; }")
[void]$sb.AppendLine(".value { font-size: 10px; font-weight: 700; fill: #222; }")
[void]$sb.AppendLine(".bar.masking { fill: #3569a8; }")
[void]$sb.AppendLine(".bar.transform { fill: #d67b35; }")
[void]$sb.AppendLine(".legend-text { font-size: 13px; fill: #333; }")
[void]$sb.AppendLine("</style>")
[void]$sb.AppendLine("<rect x='0' y='0' width='$width' height='$height' fill='#ffffff'/>")
[void]$sb.AppendLine("<text x='$($width / 2)' y='35' class='title' text-anchor='middle'>Masking-based vs Transformation-based Augmentation Performance</text>")
[void]$sb.AppendLine("<text x='$($width / 2)' y='58' class='subtitle' text-anchor='middle'>Research08, augmentation count = 1000, final test performance</text>")
[void]$sb.AppendLine("<rect x='475' y='70' width='16' height='16' class='bar masking'/>")
[void]$sb.AppendLine("<text x='498' y='83' class='legend-text'>Masking-based</text>")
[void]$sb.AppendLine("<rect x='620' y='70' width='16' height='16' class='bar transform'/>")
[void]$sb.AppendLine("<text x='643' y='83' class='legend-text'>Transformation-based</text>")

Add-AxisAndBars $sb $metrics[0] $left1 $top1
Add-AxisAndBars $sb $metrics[1] $left2 $top1
Add-AxisAndBars $sb $metrics[2] $left1 $top2
Add-AxisAndBars $sb $metrics[3] $left2 $top2

[void]$sb.AppendLine("</svg>")

[System.IO.File]::WriteAllText($OutPath, $sb.ToString(), [System.Text.Encoding]::UTF8)
Write-Output $OutPath
