$Root = Split-Path -Parent $PSScriptRoot
$MetricPath = Join-Path $Root "data\research02\results\tsne_distribution_comparison_metrics.csv"
$QualityPath = Join-Path $Root "data\research02\results\generated_quality_metrics.csv"
$FigureDir = Join-Path $Root "data\research02\figures"

$metrics = Import-Csv $MetricPath
$quality = Import-Csv $QualityPath
$methods = @("GT-GAN", "Diffusion", "Masking GT-GAN", "Masking Diffusion")
$colors = @{
    "GT-GAN" = "#f28e2b"
    "Diffusion" = "#e15759"
    "Masking GT-GAN" = "#59a14f"
    "Masking Diffusion" = "#4e79a7"
}

function Num($value) {
    return [double]::Parse([string]$value, [Globalization.CultureInfo]::InvariantCulture)
}

function Add-Text($sb, $x, $y, $text, $size, $anchor = "middle", $weight = "400", $fill = "#1f2933") {
    [void]$sb.AppendLine("<text x='$x' y='$y' font-family='Arial, sans-serif' font-size='$size' font-weight='$weight' text-anchor='$anchor' fill='$fill'>$text</text>")
}

function Draw-BarChart($path, $metricName, $title, $subtitle, $lowerIsBetter) {
    $rows = foreach ($method in $methods) {
        $row = $metrics | Where-Object { $_.method -eq $method } | Select-Object -First 1
        $value = if ($metricName -eq "coverage_pct") {
            (Num $row.'real_coverage_within_0.25_median_real_dist') * 100.0
        } else {
            Num $row.$metricName
        }
        [pscustomobject]@{ Method = $method; Value = $value }
    }

    $width = 920
    $height = 540
    $left = 90
    $top = 82
    $plotW = 760
    $plotH = 330
    $maxValue = ($rows | Measure-Object Value -Maximum).Maximum
    if ($maxValue -le 0) { $maxValue = 1 }
    $scaleMax = $maxValue * 1.12
    $barW = 118
    $gap = 58

    $sb = [Text.StringBuilder]::new()
    [void]$sb.AppendLine("<svg xmlns='http://www.w3.org/2000/svg' width='$width' height='$height' viewBox='0 0 $width $height'>")
    [void]$sb.AppendLine("<rect width='100%' height='100%' fill='white'/>")
    Add-Text $sb ($width / 2) 35 $title 20 "middle" "700"
    Add-Text $sb ($width / 2) 58 $subtitle 13 "middle" "400" "#52616f"

    for ($i = 0; $i -le 4; $i++) {
        $y = $top + $plotH - ($plotH * $i / 4)
        $tick = $scaleMax * $i / 4
        [void]$sb.AppendLine("<line x1='$left' y1='$y' x2='$($left + $plotW)' y2='$y' stroke='#d9e1e8' stroke-width='1'/>")
        Add-Text $sb ($left - 14) ($y + 4) $tick.ToString("0.##") 11 "end" "400" "#52616f"
    }

    [void]$sb.AppendLine("<line x1='$left' y1='$($top + $plotH)' x2='$($left + $plotW)' y2='$($top + $plotH)' stroke='#8a97a5'/>")
    [void]$sb.AppendLine("<line x1='$left' y1='$top' x2='$left' y2='$($top + $plotH)' stroke='#8a97a5'/>")

    for ($i = 0; $i -lt $rows.Count; $i++) {
        $row = $rows[$i]
        $x = $left + 62 + ($i * ($barW + $gap))
        $barH = if ($scaleMax -eq 0) { 0 } else { $plotH * $row.Value / $scaleMax }
        $y = $top + $plotH - $barH
        $color = $colors[$row.Method]
        [void]$sb.AppendLine("<rect x='$x' y='$y' width='$barW' height='$barH' fill='$color' opacity='0.82'/>")
        Add-Text $sb ($x + $barW / 2) ($y - 8) $row.Value.ToString("0.###") 12 "middle" "700"
        $label = $row.Method -replace " ", "&#10;"
        Add-Text $sb ($x + $barW / 2) ($top + $plotH + 31) $label 12 "middle" "400"
    }

    $note = if ($lowerIsBetter) { "Smaller values indicate stronger alignment with real anomaly windows." } else { "Larger values indicate broader coverage of real anomaly windows." }
    Add-Text $sb ($width / 2) 500 $note 12 "middle" "400" "#52616f"
    [void]$sb.AppendLine("</svg>")
    Set-Content -Path $path -Value $sb.ToString() -Encoding UTF8
}

function Draw-Heatmap($path) {
    $cols = @("mean_abs_diff", "std_abs_diff", "corr_abs_diff")
    $labels = @{
        "mean_abs_diff" = "Mean"
        "std_abs_diff" = "Std"
        "corr_abs_diff" = "Corr"
    }
    $values = @()
    foreach ($method in $methods) {
        $row = $quality | Where-Object { $_.method -eq $method } | Select-Object -First 1
        foreach ($col in $cols) {
            $values += Num $row.$col
        }
    }
    $maxValue = ($values | Measure-Object -Maximum).Maximum
    if ($maxValue -le 0) { $maxValue = 1 }

    $width = 820
    $height = 430
    $left = 210
    $top = 88
    $cellW = 150
    $cellH = 62

    $sb = [Text.StringBuilder]::new()
    [void]$sb.AppendLine("<svg xmlns='http://www.w3.org/2000/svg' width='$width' height='$height' viewBox='0 0 $width $height'>")
    [void]$sb.AppendLine("<rect width='100%' height='100%' fill='white'/>")
    Add-Text $sb ($width / 2) 35 "Feature-Level Distribution Difference" 20 "middle" "700"
    Add-Text $sb ($width / 2) 58 "Absolute difference between real and generated anomaly windows" 13 "middle" "400" "#52616f"

    for ($c = 0; $c -lt $cols.Count; $c++) {
        Add-Text $sb ($left + $c * $cellW + $cellW / 2) 78 $labels[$cols[$c]] 13 "middle" "700"
    }

    for ($r = 0; $r -lt $methods.Count; $r++) {
        $method = $methods[$r]
        $row = $quality | Where-Object { $_.method -eq $method } | Select-Object -First 1
        Add-Text $sb ($left - 18) ($top + $r * $cellH + 37) $method 13 "end" "700"
        for ($c = 0; $c -lt $cols.Count; $c++) {
            $col = $cols[$c]
            $value = Num $row.$col
            $intensity = [Math]::Min(1.0, $value / $maxValue)
            $red = [int](255 - 40 * (1 - $intensity))
            $green = [int](245 - 165 * $intensity)
            $blue = [int](235 - 185 * $intensity)
            $fill = "rgb($red,$green,$blue)"
            $x = $left + $c * $cellW
            $y = $top + $r * $cellH
            [void]$sb.AppendLine("<rect x='$x' y='$y' width='$cellW' height='$cellH' fill='$fill' stroke='white' stroke-width='2'/>")
            Add-Text $sb ($x + $cellW / 2) ($y + 37) $value.ToString("0.###") 13 "middle" "700"
        }
    }

    Add-Text $sb ($width / 2) 390 "Lighter cells are closer to the real anomaly distribution." 12 "middle" "400" "#52616f"
    [void]$sb.AppendLine("</svg>")
    Set-Content -Path $path -Value $sb.ToString() -Encoding UTF8
}

Draw-BarChart (Join-Path $FigureDir "research02_centroid_distance_bar.svg") "centroid_distance" "Centroid Distance to Real Anomaly Distribution" "Distribution-level shift by generation method" $true
Draw-BarChart (Join-Path $FigureDir "research02_gen_to_real_nn_bar.svg") "gen_to_real_nn_mean" "Generated-to-Real Nearest Neighbor Distance" "How far generated samples are from the closest real anomaly" $true
Draw-BarChart (Join-Path $FigureDir "research02_real_coverage_bar.svg") "coverage_pct" "Real Anomaly Coverage by Generated Samples" "Share of real anomaly windows covered by generated samples" $false
Draw-Heatmap (Join-Path $FigureDir "research02_feature_difference_heatmap.svg")

Write-Host "saved: data/research02/figures/research02_centroid_distance_bar.svg"
Write-Host "saved: data/research02/figures/research02_gen_to_real_nn_bar.svg"
Write-Host "saved: data/research02/figures/research02_real_coverage_bar.svg"
Write-Host "saved: data/research02/figures/research02_feature_difference_heatmap.svg"
