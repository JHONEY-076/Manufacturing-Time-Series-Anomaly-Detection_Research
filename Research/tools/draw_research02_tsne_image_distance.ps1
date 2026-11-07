$Root = Split-Path -Parent $PSScriptRoot
$ImagePath = Join-Path $Root "data\research02\figures\tsne_2x2_pairwise_real_vs_generated.png"
$ResultDir = Join-Path $Root "data\research02\results"
$FigureDir = Join-Path $Root "data\research02\figures"
$CsvPath = Join-Path $ResultDir "tsne_image_distance_to_real_centroid_summary.csv"
$SvgPath = Join-Path $FigureDir "research02_tsne_image_distance_to_real_centroid_bar.svg"
$RatioSvgPath = Join-Path $FigureDir "research02_tsne_radial_distance_ratio_bar.svg"
$PaperRatioSvgPath = Join-Path $FigureDir "research02_tsne_radial_distance_ratio_paper.svg"

Add-Type -AssemblyName System.Drawing

function Is-BluePoint($color) {
    return ($color.R -ge 65 -and $color.R -le 140 -and
            $color.G -ge 125 -and $color.G -le 190 -and
            $color.B -ge 165 -and $color.B -le 235)
}

function Is-OrangePoint($color) {
    return ($color.R -ge 220 -and
            $color.G -ge 120 -and $color.G -le 190 -and
            $color.B -ge 20 -and $color.B -le 120)
}

function Median($values) {
    $arr = @($values | Sort-Object)
    if ($arr.Count -eq 0) { return 0.0 }
    $mid = [int]($arr.Count / 2)
    if ($arr.Count % 2 -eq 1) { return [double]$arr[$mid] }
    return ([double]$arr[$mid - 1] + [double]$arr[$mid]) / 2.0
}

function Quantile($values, $q) {
    $arr = @($values | Sort-Object)
    if ($arr.Count -eq 0) { return 0.0 }
    $pos = ($arr.Count - 1) * $q
    $lo = [Math]::Floor($pos)
    $hi = [Math]::Ceiling($pos)
    if ($lo -eq $hi) { return [double]$arr[$lo] }
    $w = $pos - $lo
    return ([double]$arr[$lo] * (1.0 - $w)) + ([double]$arr[$hi] * $w)
}

function Add-Text($sb, $x, $y, $text, $size, $anchor = "middle", $weight = "400", $fill = "#1f2933") {
    [void]$sb.AppendLine("<text x='$x' y='$y' font-family='Arial, sans-serif' font-size='$size' font-weight='$weight' text-anchor='$anchor' fill='$fill'>$text</text>")
}

$panels = @(
    @{ Method = "GT-GAN"; Col = 0; Row = 0 },
    @{ Method = "Masking GT-GAN"; Col = 1; Row = 0 },
    @{ Method = "Diffusion"; Col = 0; Row = 1 },
    @{ Method = "Masking Diffusion"; Col = 1; Row = 1 }
)

$img = [System.Drawing.Bitmap]::FromFile($ImagePath)
$halfW = [int]($img.Width / 2)
$halfH = [int]($img.Height / 2)

# Crop only the plotted axes area inside each subplot to reduce title and label pixels.
$cropLeft = 135
$cropTop = 118
$cropRight = 1335
$cropBottom = 820
$pixelStep = 3

$rows = @()
foreach ($panel in $panels) {
    $x0 = $panel.Col * $halfW + $cropLeft
    $y0 = $panel.Row * $halfH + $cropTop
    $x1 = $panel.Col * $halfW + $cropRight
    $y1 = $panel.Row * $halfH + $cropBottom

    $blueXs = New-Object System.Collections.Generic.List[double]
    $blueYs = New-Object System.Collections.Generic.List[double]
    $orangeXs = New-Object System.Collections.Generic.List[double]
    $orangeYs = New-Object System.Collections.Generic.List[double]

    for ($y = $y0; $y -le $y1; $y += $pixelStep) {
        for ($x = $x0; $x -le $x1; $x += $pixelStep) {
            $c = $img.GetPixel($x, $y)
            if (Is-BluePoint $c) {
                $blueXs.Add([double]($x - $x0))
                $blueYs.Add([double]($y - $y0))
            } elseif (Is-OrangePoint $c) {
                $orangeXs.Add([double]($x - $x0))
                $orangeYs.Add([double]($y - $y0))
            }
        }
    }

    $cx = if ($blueXs.Count -gt 0) { ($blueXs | Measure-Object -Average).Average } else { 0.0 }
    $cy = if ($blueYs.Count -gt 0) { ($blueYs | Measure-Object -Average).Average } else { 0.0 }
    $distances = New-Object System.Collections.Generic.List[double]
    $blueDistances = New-Object System.Collections.Generic.List[double]
    for ($i = 0; $i -lt $blueXs.Count; $i++) {
        $dx = $blueXs[$i] - $cx
        $dy = $blueYs[$i] - $cy
        $blueDistances.Add([Math]::Sqrt(($dx * $dx) + ($dy * $dy)))
    }
    for ($i = 0; $i -lt $orangeXs.Count; $i++) {
        $dx = $orangeXs[$i] - $cx
        $dy = $orangeYs[$i] - $cy
        $distances.Add([Math]::Sqrt(($dx * $dx) + ($dy * $dy)))
    }

    $mean = if ($distances.Count -gt 0) { ($distances | Measure-Object -Average).Average } else { 0.0 }
    $blueMean = if ($blueDistances.Count -gt 0) { ($blueDistances | Measure-Object -Average).Average } else { 0.0 }
    $ratio = if ($blueMean -gt 0) { $mean / $blueMean } else { 0.0 }
    $rows += [pscustomobject]@{
        method = $panel.Method
        blue_pixel_count = $blueXs.Count
        orange_pixel_count = $orangeXs.Count
        real_centroid_x_px = [Math]::Round($cx, 3)
        real_centroid_y_px = [Math]::Round($cy, 3)
        real_distance_mean_px = [Math]::Round($blueMean, 3)
        real_distance_median_px = [Math]::Round((Median $blueDistances), 3)
        generated_distance_mean_px = [Math]::Round($mean, 3)
        generated_distance_median_px = [Math]::Round((Median $distances), 3)
        generated_distance_q1_px = [Math]::Round((Quantile $distances 0.25), 3)
        generated_distance_q3_px = [Math]::Round((Quantile $distances 0.75), 3)
        generated_to_real_radial_mean_ratio = [Math]::Round($ratio, 3)
        radial_mean_abs_delta_px = [Math]::Round([Math]::Abs($mean - $blueMean), 3)
    }
}
$img.Dispose()

$rows | Export-Csv -Path $CsvPath -NoTypeInformation -Encoding UTF8

$colors = @{
    "GT-GAN" = "#f28e2b"
    "Diffusion" = "#e15759"
    "Masking GT-GAN" = "#59a14f"
    "Masking Diffusion" = "#4e79a7"
}

$width = 980
$height = 570
$left = 95
$top = 85
$plotW = 790
$plotH = 335
$maxValue = ($rows | Measure-Object generated_distance_mean_px -Maximum).Maximum
if ($maxValue -le 0) { $maxValue = 1 }
$scaleMax = $maxValue * 1.18
$barW = 118
$gap = 66

$sb = [Text.StringBuilder]::new()
[void]$sb.AppendLine("<svg xmlns='http://www.w3.org/2000/svg' width='$width' height='$height' viewBox='0 0 $width $height'>")
[void]$sb.AppendLine("<rect width='100%' height='100%' fill='white'/>")
Add-Text $sb ($width / 2) 35 "t-SNE Image-Derived Distance to Real-Anomaly Centroid" 20 "middle" "700"
Add-Text $sb ($width / 2) 59 "Generated point distance from the real-data center in each t-SNE subplot" 13 "middle" "400" "#52616f"

for ($i = 0; $i -le 4; $i++) {
    $y = $top + $plotH - ($plotH * $i / 4)
    $tick = $scaleMax * $i / 4
    [void]$sb.AppendLine("<line x1='$left' y1='$y' x2='$($left + $plotW)' y2='$y' stroke='#d9e1e8' stroke-width='1'/>")
    Add-Text $sb ($left - 14) ($y + 4) $tick.ToString("0") 11 "end" "400" "#52616f"
}
[void]$sb.AppendLine("<line x1='$left' y1='$($top + $plotH)' x2='$($left + $plotW)' y2='$($top + $plotH)' stroke='#8a97a5'/>")
[void]$sb.AppendLine("<line x1='$left' y1='$top' x2='$left' y2='$($top + $plotH)' stroke='#8a97a5'/>")

for ($i = 0; $i -lt $rows.Count; $i++) {
    $row = $rows[$i]
    $x = $left + 62 + ($i * ($barW + $gap))
    $barH = $plotH * $row.generated_distance_mean_px / $scaleMax
    $y = $top + $plotH - $barH
    $color = $colors[$row.method]
    $q1Y = $top + $plotH - ($plotH * $row.generated_distance_q1_px / $scaleMax)
    $q3Y = $top + $plotH - ($plotH * $row.generated_distance_q3_px / $scaleMax)
    [void]$sb.AppendLine("<rect x='$x' y='$y' width='$barW' height='$barH' fill='$color' opacity='0.82'/>")
    [void]$sb.AppendLine("<line x1='$($x + $barW / 2)' y1='$q1Y' x2='$($x + $barW / 2)' y2='$q3Y' stroke='#1f2933' stroke-width='2'/>")
    [void]$sb.AppendLine("<line x1='$($x + $barW / 2 - 15)' y1='$q1Y' x2='$($x + $barW / 2 + 15)' y2='$q1Y' stroke='#1f2933' stroke-width='2'/>")
    [void]$sb.AppendLine("<line x1='$($x + $barW / 2 - 15)' y1='$q3Y' x2='$($x + $barW / 2 + 15)' y2='$q3Y' stroke='#1f2933' stroke-width='2'/>")
    Add-Text $sb ($x + $barW / 2) ($y - 8) $row.generated_distance_mean_px.ToString("0.0") 12 "middle" "700"
    $label = $row.method -replace " ", "&#10;"
    Add-Text $sb ($x + $barW / 2) ($top + $plotH + 31) $label 12 "middle" "400"
}

Add-Text $sb ($width / 2) 510 "Bars show mean pixel distance; whiskers show Q1-Q3. Use as a visual diagnostic, not as the final numeric metric." 12 "middle" "400" "#52616f"
[void]$sb.AppendLine("</svg>")
Set-Content -Path $SvgPath -Value $sb.ToString() -Encoding UTF8

Write-Host "saved: data/research02/results/tsne_image_distance_to_real_centroid_summary.csv"
Write-Host "saved: data/research02/figures/research02_tsne_image_distance_to_real_centroid_bar.svg"

$ratioWidth = 980
$ratioHeight = 560
$ratioLeft = 95
$ratioTop = 85
$ratioPlotW = 790
$ratioPlotH = 330
$ratioMax = ($rows | Measure-Object generated_to_real_radial_mean_ratio -Maximum).Maximum
if ($ratioMax -lt 1.1) { $ratioMax = 1.1 }
$ratioScaleMax = $ratioMax * 1.18

$sb = [Text.StringBuilder]::new()
[void]$sb.AppendLine("<svg xmlns='http://www.w3.org/2000/svg' width='$ratioWidth' height='$ratioHeight' viewBox='0 0 $ratioWidth $ratioHeight'>")
[void]$sb.AppendLine("<rect width='100%' height='100%' fill='white'/>")
Add-Text $sb ($ratioWidth / 2) 35 "t-SNE Radial Distance Ratio" 20 "middle" "700"
Add-Text $sb ($ratioWidth / 2) 59 "Generated mean distance from real centroid divided by real mean distance" 13 "middle" "400" "#52616f"

for ($i = 0; $i -le 4; $i++) {
    $y = $ratioTop + $ratioPlotH - ($ratioPlotH * $i / 4)
    $tick = $ratioScaleMax * $i / 4
    [void]$sb.AppendLine("<line x1='$ratioLeft' y1='$y' x2='$($ratioLeft + $ratioPlotW)' y2='$y' stroke='#d9e1e8' stroke-width='1'/>")
    Add-Text $sb ($ratioLeft - 14) ($y + 4) $tick.ToString("0.00") 11 "end" "400" "#52616f"
}
$oneY = $ratioTop + $ratioPlotH - ($ratioPlotH * 1.0 / $ratioScaleMax)
[void]$sb.AppendLine("<line x1='$ratioLeft' y1='$oneY' x2='$($ratioLeft + $ratioPlotW)' y2='$oneY' stroke='#1f2933' stroke-width='1.5' stroke-dasharray='5 5'/>")
Add-Text $sb ($ratioLeft + $ratioPlotW + 8) ($oneY + 4) "1.0" 11 "start" "700" "#1f2933"
[void]$sb.AppendLine("<line x1='$ratioLeft' y1='$($ratioTop + $ratioPlotH)' x2='$($ratioLeft + $ratioPlotW)' y2='$($ratioTop + $ratioPlotH)' stroke='#8a97a5'/>")
[void]$sb.AppendLine("<line x1='$ratioLeft' y1='$ratioTop' x2='$ratioLeft' y2='$($ratioTop + $ratioPlotH)' stroke='#8a97a5'/>")

for ($i = 0; $i -lt $rows.Count; $i++) {
    $row = $rows[$i]
    $x = $ratioLeft + 62 + ($i * ($barW + $gap))
    $barH = $ratioPlotH * $row.generated_to_real_radial_mean_ratio / $ratioScaleMax
    $y = $ratioTop + $ratioPlotH - $barH
    $color = $colors[$row.method]
    [void]$sb.AppendLine("<rect x='$x' y='$y' width='$barW' height='$barH' fill='$color' opacity='0.82'/>")
    Add-Text $sb ($x + $barW / 2) ($y - 8) $row.generated_to_real_radial_mean_ratio.ToString("0.00") 12 "middle" "700"
    $label = $row.method -replace " ", "&#10;"
    Add-Text $sb ($x + $barW / 2) ($ratioTop + $ratioPlotH + 31) $label 12 "middle" "400"
}

Add-Text $sb ($ratioWidth / 2) 503 "A value near 1 means generated samples have a similar radial spread around the real-data center." 12 "middle" "400" "#52616f"
[void]$sb.AppendLine("</svg>")
Set-Content -Path $RatioSvgPath -Value $sb.ToString() -Encoding UTF8
Write-Host "saved: data/research02/figures/research02_tsne_radial_distance_ratio_bar.svg"

$paperRows = foreach ($name in @("GT-GAN", "Diffusion", "Masking GT-GAN", "Masking Diffusion")) {
    $rows | Where-Object { $_.method -eq $name } | Select-Object -First 1
}
$paperLabels = @{
    "GT-GAN" = "GT-GAN"
    "Diffusion" = "Diffusion"
    "Masking GT-GAN" = "Mask-GTGAN"
    "Masking Diffusion" = "Mask-Diffusion"
}
$paperColors = @{
    "GT-GAN" = "#d77a1f"
    "Diffusion" = "#c94c4c"
    "Masking GT-GAN" = "#4f8f45"
    "Masking Diffusion" = "#3569a8"
}

$paperWidth = 860
$paperHeight = 520
$paperLeft = 92
$paperTop = 70
$paperPlotW = 690
$paperPlotH = 320
$paperMax = 3.35
$paperBarW = 78
$paperGap = 82

$sb = [Text.StringBuilder]::new()
[void]$sb.AppendLine("<svg xmlns='http://www.w3.org/2000/svg' width='$paperWidth' height='$paperHeight' viewBox='0 0 $paperWidth $paperHeight'>")
[void]$sb.AppendLine("<rect width='100%' height='100%' fill='#ffffff'/>")
Add-Text $sb ($paperWidth / 2) 34 "t-SNE Radial Distance from Real-Anomaly Center" 19 "middle" "700" "#17212b"
Add-Text $sb ($paperWidth / 2) 57 "Generated spread normalized by the real anomaly spread" 12 "middle" "400" "#52616f"

for ($i = 0; $i -le 6; $i++) {
    $tick = $i * 0.5
    $y = $paperTop + $paperPlotH - ($paperPlotH * $tick / $paperMax)
    [void]$sb.AppendLine("<line x1='$paperLeft' y1='$y' x2='$($paperLeft + $paperPlotW)' y2='$y' stroke='#e5eaf0' stroke-width='1'/>")
    Add-Text $sb ($paperLeft - 12) ($y + 4) $tick.ToString("0.0") 11 "end" "400" "#52616f"
}

$baselineY = $paperTop + $paperPlotH - ($paperPlotH / $paperMax)
[void]$sb.AppendLine("<line x1='$paperLeft' y1='$baselineY' x2='$($paperLeft + $paperPlotW)' y2='$baselineY' stroke='#17212b' stroke-width='1.4' stroke-dasharray='6 5'/>")
Add-Text $sb ($paperLeft + $paperPlotW - 52) ($baselineY - 7) "Real spread = 1.0" 11 "start" "700" "#17212b"

[void]$sb.AppendLine("<line x1='$paperLeft' y1='$($paperTop + $paperPlotH)' x2='$($paperLeft + $paperPlotW)' y2='$($paperTop + $paperPlotH)' stroke='#9aa8b6' stroke-width='1'/>")
[void]$sb.AppendLine("<line x1='$paperLeft' y1='$paperTop' x2='$paperLeft' y2='$($paperTop + $paperPlotH)' stroke='#9aa8b6' stroke-width='1'/>")

[void]$sb.AppendLine("<g transform='translate(22,$($paperTop + 170)) rotate(-90)'><text font-family='Arial, sans-serif' font-size='12' font-weight='700' text-anchor='middle' fill='#334155'>Generated / real distance</text></g>")

for ($i = 0; $i -lt $paperRows.Count; $i++) {
    $row = $paperRows[$i]
    $ratio = [double]$row.generated_to_real_radial_mean_ratio
    $x = $paperLeft + 66 + ($i * ($paperBarW + $paperGap))
    $barH = $paperPlotH * $ratio / $paperMax
    $y = $paperTop + $paperPlotH - $barH
    $color = $paperColors[$row.method]
    [void]$sb.AppendLine("<rect x='$x' y='$y' width='$paperBarW' height='$barH' rx='2' fill='$color' opacity='0.9'/>")
    Add-Text $sb ($x + $paperBarW / 2) ($y - 9) $ratio.ToString("0.00") 12 "middle" "700" "#17212b"
    Add-Text $sb ($x + $paperBarW / 2) ($paperTop + $paperPlotH + 28) $paperLabels[$row.method] 12 "middle" "700" "#334155"
}

Add-Text $sb ($paperWidth / 2) 458 "Ratios closer to 1 indicate generated samples follow the radial spread of real anomaly windows." 12 "middle" "400" "#52616f"
Add-Text $sb ($paperWidth / 2) 480 "Plain generation drifts outward, while masking-based generation remains near the real anomaly manifold." 12 "middle" "400" "#52616f"
[void]$sb.AppendLine("</svg>")
Set-Content -Path $PaperRatioSvgPath -Value $sb.ToString() -Encoding UTF8
Write-Host "saved: data/research02/figures/research02_tsne_radial_distance_ratio_paper.svg"
