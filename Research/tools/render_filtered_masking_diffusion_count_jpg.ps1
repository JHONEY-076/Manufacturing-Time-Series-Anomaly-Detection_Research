Add-Type -AssemblyName System.Drawing

$outPath = Join-Path (Get-Location) "pictures\filtered_masking_diffusion_count_performance.jpg"

$scale = 2
$width = 960 * $scale
$height = 620 * $scale
$bmp = New-Object System.Drawing.Bitmap($width, $height)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$g.Clear([System.Drawing.Color]::White)
$g.ScaleTransform($scale, $scale)

$font = New-Object System.Drawing.Font("Arial", 12)
$axisFont = New-Object System.Drawing.Font("Arial", 15)
$legendFont = New-Object System.Drawing.Font("Arial", 12)
$black = [System.Drawing.Brushes]::Black
$gridPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(215, 222, 232), 1)
$axisPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(17, 24, 39), 1.4)
$tickPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(170, 180, 193), 1)

$left = 96.0
$top = 42.0
$plotW = 720.0
$plotH = 470.0
$bottom = $top + $plotH

$counts = @(50, 100, 200, 500, 750, 1000)
$xPos = @(136.0, 264.0, 392.0, 520.0, 648.0, 776.0)
$yMin = 0.50
$yMax = 1.00

function YPos([double]$v) {
    return $bottom - (($v - $yMin) / ($yMax - $yMin) * $plotH)
}

function DrawCenteredText($text, [double]$x, [double]$y, $drawFont) {
    $size = $g.MeasureString($text, $drawFont)
    $g.DrawString($text, $drawFont, $black, [float]($x - $size.Width / 2), [float]$y)
}

function DrawRightText($text, [double]$x, [double]$y, $drawFont) {
    $size = $g.MeasureString($text, $drawFont)
    $g.DrawString($text, $drawFont, $black, [float]($x - $size.Width), [float]$y)
}

foreach ($tick in @(0.50, 0.60, 0.70, 0.80, 0.90, 1.00)) {
    $y = YPos $tick
    $g.DrawLine($gridPen, [float]$left, [float]$y, [float]($left + $plotW), [float]$y)
    DrawRightText ($tick.ToString("0.00")) 82 ($y - 8) $font
}

$g.DrawLine($axisPen, [float]$left, [float]$bottom, [float]($left + $plotW), [float]$bottom)
$g.DrawLine($axisPen, [float]$left, [float]$top, [float]$left, [float]$bottom)

for ($i = 0; $i -lt $counts.Count; $i++) {
    $x = $xPos[$i]
    $g.DrawLine($tickPen, [float]$x, [float]$bottom, [float]$x, [float]($bottom + 6))
    DrawCenteredText ([string]$counts[$i]) $x 528 $font
}

DrawCenteredText "Number of generated anomaly windows" 456 575 $axisFont

$state = $g.Save()
$g.TranslateTransform(28, 277)
$g.RotateTransform(-90)
DrawCenteredText "Final test score" 0 -8 $axisFont
$g.Restore($state)

$series = @(
    @{ Name = "Precision"; Color = [System.Drawing.Color]::FromArgb(31, 119, 180); Values = @(0.7922, 0.8062, 0.7725, 0.8152, 0.9522, 0.8333) },
    @{ Name = "Recall"; Color = [System.Drawing.Color]::FromArgb(44, 160, 44); Values = @(0.6182, 0.6182, 0.6081, 0.7601, 0.8074, 0.8446) },
    @{ Name = "F1-score"; Color = [System.Drawing.Color]::FromArgb(214, 39, 40); Values = @(0.6945, 0.6998, 0.6805, 0.7867, 0.8739, 0.8389) },
    @{ Name = "F2-score"; Color = [System.Drawing.Color]::FromArgb(148, 103, 189); Values = @(0.6466, 0.6485, 0.6351, 0.7705, 0.8328, 0.8423) },
    @{ Name = "AUPRC"; Color = [System.Drawing.Color]::FromArgb(255, 127, 14); Values = @(0.7872, 0.7899, 0.7774, 0.8836, 0.9432, 0.9221) }
)

foreach ($s in $series) {
    $pen = New-Object System.Drawing.Pen($s.Color, 3.2)
    $points = New-Object "System.Drawing.PointF[]" $xPos.Count
    for ($i = 0; $i -lt $xPos.Count; $i++) {
        $points[$i] = New-Object System.Drawing.PointF([float]$xPos[$i], [float](YPos $s.Values[$i]))
    }
    $g.DrawLines($pen, $points)
    $brush = New-Object System.Drawing.SolidBrush($s.Color)
    foreach ($p in $points) {
        $g.FillEllipse($brush, $p.X - 4.8, $p.Y - 4.8, 9.6, 9.6)
        $g.DrawEllipse([System.Drawing.Pens]::White, $p.X - 4.8, $p.Y - 4.8, 9.6, 9.6)
    }
    $pen.Dispose()
    $brush.Dispose()
}

$legendX = 640.0
$legendY = 322.0
$legendW = 154.0
$legendH = 132.0
$legendPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(209, 213, 219), 1)
$g.FillRectangle([System.Drawing.Brushes]::White, [float]$legendX, [float]$legendY, [float]$legendW, [float]$legendH)
$g.DrawRectangle($legendPen, [float]$legendX, [float]$legendY, [float]$legendW, [float]$legendH)

for ($i = 0; $i -lt $series.Count; $i++) {
    $y = 346 + 23 * $i
    $s = $series[$i]
    $pen = New-Object System.Drawing.Pen($s.Color, 3.2)
    $brush = New-Object System.Drawing.SolidBrush($s.Color)
    $g.DrawLine($pen, 658, $y, 688, $y)
    $g.FillEllipse($brush, 668.8, $y - 4.2, 8.4, 8.4)
    $g.DrawString($s.Name, $legendFont, $black, 698, $y - 8)
    $pen.Dispose()
    $brush.Dispose()
}

$encoder = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq "image/jpeg" }
$params = New-Object System.Drawing.Imaging.EncoderParameters(1)
$params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, 95L)
$bmp.Save($outPath, $encoder, $params)

$params.Dispose()
$g.Dispose()
$bmp.Dispose()

Write-Output $outPath
