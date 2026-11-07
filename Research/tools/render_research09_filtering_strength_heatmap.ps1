Add-Type -AssemblyName System.Drawing

$outPath = Join-Path (Get-Location) "pictures\research09_filtering_strength_heatmap.jpg"

$scale = 2
$width = 1200 * $scale
$height = 560 * $scale
$bmp = New-Object System.Drawing.Bitmap($width, $height)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$g.Clear([System.Drawing.Color]::White)
$g.ScaleTransform($scale, $scale)

$font = New-Object System.Drawing.Font("Arial", 13)
$smallFont = New-Object System.Drawing.Font("Arial", 11)
$boldFont = New-Object System.Drawing.Font("Arial", 13, [System.Drawing.FontStyle]::Bold)
$black = [System.Drawing.Brushes]::Black
$borderPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(210, 216, 225), 1)

$rows = @(
    @{ Condition = "750 / loose"; Precision = 0.8490; Recall = 0.8547; F1 = 0.8519; F2 = 0.8536; AUPRC = 0.9389; FN = 43; FP = 45 },
    @{ Condition = "750 / default"; Precision = 0.8911; Recall = 0.7736; F1 = 0.8282; F2 = 0.7946; AUPRC = 0.9276; FN = 67; FP = 28 },
    @{ Condition = "750 / strict"; Precision = 0.8817; Recall = 0.7804; F1 = 0.8280; F2 = 0.7988; AUPRC = 0.9070; FN = 65; FP = 31 },
    @{ Condition = "1000 / loose"; Precision = 0.9945; Recall = 0.6149; F1 = 0.7599; F2 = 0.6657; AUPRC = 0.9419; FN = 114; FP = 1 },
    @{ Condition = "1000 / default"; Precision = 0.9333; Recall = 0.6622; F1 = 0.7747; F2 = 0.7030; AUPRC = 0.9093; FN = 100; FP = 14 },
    @{ Condition = "1000 / strict"; Precision = 0.9038; Recall = 0.7939; F1 = 0.8453; F2 = 0.8137; AUPRC = 0.9326; FN = 61; FP = 25 }
)

$cols = @("Precision", "Recall", "F1", "F2", "AUPRC", "FN", "FP")
$higherBetter = @("Precision", "Recall", "F1", "F2", "AUPRC")
$lowerBetter = @("FN", "FP")

$left = 70
$top = 70
$labelW = 180
$cellW = 120
$cellH = 58
$headerH = 50

function Lerp([int]$a, [int]$b, [double]$t) {
    return [int][Math]::Round($a + ($b - $a) * $t)
}

function HeatColor([double]$t, [bool]$blue) {
    $t = [Math]::Max(0, [Math]::Min(1, $t))
    if ($blue) {
        return [System.Drawing.Color]::FromArgb((Lerp 239 37 $t), (Lerp 246 99 $t), (Lerp 255 235 $t))
    }
    return [System.Drawing.Color]::FromArgb((Lerp 254 220 $t), (Lerp 242 38 $t), (Lerp 242 38 $t))
}

function DrawCentered($text, [double]$x, [double]$y, [double]$w, [double]$h, $drawFont, $brush) {
    $sf = New-Object System.Drawing.StringFormat
    $sf.Alignment = [System.Drawing.StringAlignment]::Center
    $sf.LineAlignment = [System.Drawing.StringAlignment]::Center
    $rect = New-Object System.Drawing.RectangleF([float]$x, [float]$y, [float]$w, [float]$h)
    $g.DrawString($text, $drawFont, $brush, $rect, $sf)
    $sf.Dispose()
}

$mins = @{}
$maxs = @{}
foreach ($col in $cols) {
    $vals = $rows | ForEach-Object { [double]$_[$col] }
    $mins[$col] = ($vals | Measure-Object -Minimum).Minimum
    $maxs[$col] = ($vals | Measure-Object -Maximum).Maximum
}

$headerBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(248, 250, 252))
$g.FillRectangle($headerBrush, $left, $top, $labelW + $cellW * $cols.Count, $headerH)
$g.DrawRectangle($borderPen, $left, $top, $labelW + $cellW * $cols.Count, $headerH)
DrawCentered "Condition" $left $top $labelW $headerH $boldFont $black

for ($j = 0; $j -lt $cols.Count; $j++) {
    $x = $left + $labelW + $cellW * $j
    $g.DrawRectangle($borderPen, $x, $top, $cellW, $headerH)
    DrawCentered $cols[$j] $x $top $cellW $headerH $boldFont $black
}

for ($i = 0; $i -lt $rows.Count; $i++) {
    $y = $top + $headerH + $cellH * $i
    $conditionBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 255, 255))
    $g.FillRectangle($conditionBrush, $left, $y, $labelW, $cellH)
    $g.DrawRectangle($borderPen, $left, $y, $labelW, $cellH)
    DrawCentered $rows[$i].Condition $left $y $labelW $cellH $font $black
    $conditionBrush.Dispose()

    for ($j = 0; $j -lt $cols.Count; $j++) {
        $col = $cols[$j]
        $value = [double]$rows[$i][$col]
        $range = [double]($maxs[$col] - $mins[$col])
        $raw = if ($range -eq 0) { 0.5 } else { ($value - $mins[$col]) / $range }
        $score = if ($lowerBetter -contains $col) { 1.0 - $raw } else { $raw }
        $isBlue = $higherBetter -contains $col
        $color = HeatColor $score $isBlue
        $brush = New-Object System.Drawing.SolidBrush($color)
        $x = $left + $labelW + $cellW * $j
        $g.FillRectangle($brush, $x, $y, $cellW, $cellH)
        $g.DrawRectangle($borderPen, $x, $y, $cellW, $cellH)
        $text = if ($col -in @("FN", "FP")) { [string][int]$value } else { $value.ToString("0.0000") }
        DrawCentered $text $x $y $cellW $cellH $font $black
        $brush.Dispose()
    }
}

$noteFont = New-Object System.Drawing.Font("Arial", 11)
$noteBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(75, 85, 99))
$g.DrawString("Blue cells: higher is better. Red cells: lower is better for FN and FP.", $noteFont, $noteBrush, 70, 480)

$encoder = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq "image/jpeg" }
$params = New-Object System.Drawing.Imaging.EncoderParameters(1)
$params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, 95L)
$bmp.Save($outPath, $encoder, $params)

$params.Dispose()
$g.Dispose()
$bmp.Dispose()
Write-Output $outPath
