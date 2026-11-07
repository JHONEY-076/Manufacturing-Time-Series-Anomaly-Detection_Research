$Root = Split-Path -Parent $PSScriptRoot
$InputPath = Join-Path $Root "data\research02\figures\tsne_2x2_pairwise_real_vs_generated.png"
$OutputPath = Join-Path $Root "data\research02\figures\tsne_2x2_pairwise_real_vs_generated_large_points.png"

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

$src = [System.Drawing.Bitmap]::FromFile($InputPath)
$bmp = New-Object System.Drawing.Bitmap $src.Width, $src.Height
$graphics = [System.Drawing.Graphics]::FromImage($bmp)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.DrawImage($src, 0, 0, $src.Width, $src.Height)

$orangeBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(150, 250, 164, 58))
$blueBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(180, 93, 165, 218))
$orangePen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(230, 255, 255, 255)), 1.0
$bluePen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(210, 47, 111, 159)), 0.8

$halfW = [int]($src.Width / 2)
$halfH = [int]($src.Height / 2)
$cropLeft = 135
$cropTop = 118
$cropRight = 1335
$cropBottom = 820
$pixelStep = 5

for ($panelRow = 0; $panelRow -lt 2; $panelRow++) {
    for ($panelCol = 0; $panelCol -lt 2; $panelCol++) {
        $x0 = $panelCol * $halfW + $cropLeft
        $y0 = $panelRow * $halfH + $cropTop
        $x1 = $panelCol * $halfW + $cropRight
        $y1 = $panelRow * $halfH + $cropBottom

        for ($y = $y0; $y -le $y1; $y += $pixelStep) {
            for ($x = $x0; $x -le $x1; $x += $pixelStep) {
                $c = $src.GetPixel($x, $y)
                if (Is-OrangePoint $c) {
                    $graphics.FillEllipse($orangeBrush, $x - 4, $y - 4, 8, 8)
                    $graphics.DrawEllipse($orangePen, $x - 4, $y - 4, 8, 8)
                } elseif (Is-BluePoint $c) {
                    $graphics.FillEllipse($blueBrush, $x - 3.5, $y - 3.5, 7, 7)
                    $graphics.DrawEllipse($bluePen, $x - 3.5, $y - 3.5, 7, 7)
                }
            }
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

Write-Host "saved: data/research02/figures/tsne_2x2_pairwise_real_vs_generated_large_points.png"
