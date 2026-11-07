$Root = Split-Path -Parent $PSScriptRoot
$InputPath = Join-Path $Root "data\research02\figures\tsne_2x2_pairwise_real_vs_generated.png"
$OutputPath = Join-Path $Root "data\research02\figures\tsne_2x2_pairwise_real_vs_generated_clean_large_markers.png"

Add-Type -AssemblyName System.Drawing

function Is-BluePoint($color) {
    return ($color.R -ge 65 -and $color.R -le 150 -and
            $color.G -ge 120 -and $color.G -le 200 -and
            $color.B -ge 160 -and $color.B -le 245)
}

function Is-OrangePoint($color) {
    return ($color.R -ge 215 -and
            $color.G -ge 115 -and $color.G -le 200 -and
            $color.B -ge 15 -and $color.B -le 135)
}

function Add-ClusterPoint($clusters, [double]$x, [double]$y, [double]$radius) {
    $best = $null
    $bestD2 = $radius * $radius
    foreach ($cluster in $clusters) {
        $dx = $cluster.Cx - $x
        $dy = $cluster.Cy - $y
        $d2 = ($dx * $dx) + ($dy * $dy)
        if ($d2 -lt $bestD2) {
            $bestD2 = $d2
            $best = $cluster
        }
    }
    if ($null -eq $best) {
        $clusters.Add([pscustomobject]@{ Cx = $x; Cy = $y; Count = 1 }) | Out-Null
    } else {
        $n = [double]$best.Count
        $best.Cx = (($best.Cx * $n) + $x) / ($n + 1.0)
        $best.Cy = (($best.Cy * $n) + $y) / ($n + 1.0)
        $best.Count += 1
    }
}

$src = [System.Drawing.Bitmap]::FromFile($InputPath)
$bmp = New-Object System.Drawing.Bitmap $src.Width, $src.Height
$graphics = [System.Drawing.Graphics]::FromImage($bmp)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.DrawImage($src, 0, 0, $src.Width, $src.Height)

$orangeBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(185, 250, 164, 58))
$blueBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(205, 93, 165, 218))
$orangePen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(245, 255, 255, 255)), 1.2
$bluePen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(235, 47, 111, 159)), 0.9

$halfW = [int]($src.Width / 2)
$halfH = [int]($src.Height / 2)
$cropLeft = 135
$cropTop = 118
$cropRight = 1335
$cropBottom = 820
$pixelStep = 2
$clusterRadius = 8.0

for ($panelRow = 0; $panelRow -lt 2; $panelRow++) {
    for ($panelCol = 0; $panelCol -lt 2; $panelCol++) {
        $x0 = $panelCol * $halfW + $cropLeft
        $y0 = $panelRow * $halfH + $cropTop
        $x1 = $panelCol * $halfW + $cropRight
        $y1 = $panelRow * $halfH + $cropBottom

        $orangeClusters = New-Object System.Collections.ArrayList
        $blueClusters = New-Object System.Collections.ArrayList

        for ($y = $y0; $y -le $y1; $y += $pixelStep) {
            for ($x = $x0; $x -le $x1; $x += $pixelStep) {
                $c = $src.GetPixel($x, $y)
                if (Is-OrangePoint $c) {
                    Add-ClusterPoint $orangeClusters $x $y $clusterRadius
                } elseif (Is-BluePoint $c) {
                    Add-ClusterPoint $blueClusters $x $y $clusterRadius
                }
            }
        }

        foreach ($cluster in $orangeClusters) {
            if ($cluster.Count -lt 2) { continue }
            $d = 10.5
            $graphics.FillEllipse($orangeBrush, [float]($cluster.Cx - $d / 2), [float]($cluster.Cy - $d / 2), [float]$d, [float]$d)
            $graphics.DrawEllipse($orangePen, [float]($cluster.Cx - $d / 2), [float]($cluster.Cy - $d / 2), [float]$d, [float]$d)
        }
        foreach ($cluster in $blueClusters) {
            if ($cluster.Count -lt 2) { continue }
            $d = 9.0
            $graphics.FillEllipse($blueBrush, [float]($cluster.Cx - $d / 2), [float]($cluster.Cy - $d / 2), [float]$d, [float]$d)
            $graphics.DrawEllipse($bluePen, [float]($cluster.Cx - $d / 2), [float]($cluster.Cy - $d / 2), [float]$d, [float]$d)
        }
    }
}

$bmp.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)

$orangePen.Dispose()
$bluePen.Dispose()
$orangeBrush.Dispose()
$blueBrush.Dispose()
$graphics.Dispose()
$bmp.Dispose()
$src.Dispose()

Write-Host "saved: data/research02/figures/tsne_2x2_pairwise_real_vs_generated_clean_large_markers.png"
