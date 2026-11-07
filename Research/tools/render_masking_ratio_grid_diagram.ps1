Add-Type -AssemblyName System.Drawing

$outPath = Join-Path (Get-Location) "pictures\research10_masking_ratio_grid_diagram.jpg"

$scale = 2
$width = 1500 * $scale
$height = 540 * $scale
$bmp = New-Object System.Drawing.Bitmap($width, $height)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$g.Clear([System.Drawing.Color]::White)
$g.ScaleTransform($scale, $scale)

$font = New-Object System.Drawing.Font("Arial", 12)
$smallFont = New-Object System.Drawing.Font("Arial", 10)
$titleFont = New-Object System.Drawing.Font("Arial", 15, [System.Drawing.FontStyle]::Bold)
$black = [System.Drawing.Brushes]::Black
$gridPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(189, 197, 209), 0.8)
$outlinePen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(31, 41, 55), 1.2)
$maskBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(59, 130, 246))
$crossBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(147, 197, 253))
$bgBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(248, 250, 252))
$labelBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(55, 65, 81))

function DrawCentered($text, [double]$x, [double]$y, [double]$w, [double]$h, $drawFont, $brush) {
    $sf = New-Object System.Drawing.StringFormat
    $sf.Alignment = [System.Drawing.StringAlignment]::Center
    $sf.LineAlignment = [System.Drawing.StringAlignment]::Center
    $rect = New-Object System.Drawing.RectangleF([float]$x, [float]$y, [float]$w, [float]$h)
    $g.DrawString($text, $drawFont, $brush, $rect, $sf)
    $sf.Dispose()
}

function DrawGrid($x0, $y0, $title, $subtitle, $blockLen, $groupLen, $timeStart, $featureStart) {
    $cellW = 11
    $cellH = 32
    $cols = 30
    $rows = 5
    $gridW = $cols * $cellW
    $gridH = $rows * $cellH

    $g.DrawString($title, $titleFont, $black, [float]$x0, [float]($y0 - 64))
    $g.DrawString($subtitle, $font, $labelBrush, [float]$x0, [float]($y0 - 38))

    $g.FillRectangle($bgBrush, [float]$x0, [float]$y0, [float]$gridW, [float]$gridH)

    for ($c = $timeStart; $c -lt ($timeStart + $blockLen); $c++) {
        $g.FillRectangle($crossBrush, [float]($x0 + $c * $cellW), [float]$y0, [float]$cellW, [float]$gridH)
    }
    for ($r = $featureStart; $r -lt ($featureStart + $groupLen); $r++) {
        $g.FillRectangle($maskBrush, [float]$x0, [float]($y0 + $r * $cellH), [float]$gridW, [float]$cellH)
    }
    for ($r = $featureStart; $r -lt ($featureStart + $groupLen); $r++) {
        for ($c = $timeStart; $c -lt ($timeStart + $blockLen); $c++) {
            $g.FillRectangle($maskBrush, [float]($x0 + $c * $cellW), [float]($y0 + $r * $cellH), [float]$cellW, [float]$cellH)
        }
    }

    for ($c = 0; $c -le $cols; $c++) {
        $x = $x0 + $c * $cellW
        $g.DrawLine($gridPen, [float]$x, [float]$y0, [float]$x, [float]($y0 + $gridH))
    }
    for ($r = 0; $r -le $rows; $r++) {
        $y = $y0 + $r * $cellH
        $g.DrawLine($gridPen, [float]$x0, [float]$y, [float]($x0 + $gridW), [float]$y)
    }
    $g.DrawRectangle($outlinePen, [float]$x0, [float]$y0, [float]$gridW, [float]$gridH)

    $g.DrawString("time steps = 30", $smallFont, $labelBrush, [float]($x0 + 104), [float]($y0 + $gridH + 12))
    $state = $g.Save()
    $g.TranslateTransform([float]($x0 - 44), [float]($y0 + 108))
    $g.RotateTransform(-90)
    $g.DrawString("features = 5", $smallFont, $labelBrush, 0, 0)
    $g.Restore($state)

    $area = $blockLen * $groupLen
    DrawCentered ("masked block: {0} x {1} = {2} cells" -f $blockLen, $groupLen, $area) $x0 ($y0 + $gridH + 38) $gridW 22 $font $black
}

DrawGrid 100 150 "Small mask" "temporal 0.15 -> 4 steps, feature 0.20 -> 1 feature" 4 1 22 1
DrawGrid 585 150 "Default mask" "temporal 0.25 -> 8 steps, feature 0.40 -> 2 features" 8 2 20 1
DrawGrid 1070 150 "Large mask" "temporal 0.35 -> 10 steps, feature 0.60 -> 3 features" 10 3 18 1

$legendX = 520
$legendY = 455
$g.FillRectangle($crossBrush, $legendX, $legendY, 24, 16)
$g.DrawRectangle($outlinePen, $legendX, $legendY, 24, 16)
$g.DrawString("masked time block", $font, $labelBrush, $legendX + 34, $legendY - 1)
$g.FillRectangle($maskBrush, $legendX + 205, $legendY, 24, 16)
$g.DrawRectangle($outlinePen, $legendX + 205, $legendY, 24, 16)
$g.DrawString("masked feature group", $font, $labelBrush, $legendX + 239, $legendY - 1)

$encoder = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq "image/jpeg" }
$params = New-Object System.Drawing.Imaging.EncoderParameters(1)
$params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, 95L)
$bmp.Save($outPath, $encoder, $params)

$params.Dispose()
$g.Dispose()
$bmp.Dispose()
Write-Output $outPath
