"""Create an editable diagrams.net figure and matching SVG/PNG previews."""
from pathlib import Path
import math
import random
import xml.etree.ElementTree as ET
from html import escape
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
W, H, SCALE = 1500, 550, 3
INK, BLUE, ORANGE = '#202B36', '#276A9B', '#B86B31'
GRAY, GRID = '#6A7783', '#E5EAF0'
im = Image.new('RGB', (W*SCALE, H*SCALE), 'white')
draw = ImageDraw.Draw(im)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', '<rect width="100%" height="100%" fill="white"/>']
mxfile = ET.Element('mxfile', {'host':'app.diagrams.net', 'version':'24.7.17', 'type':'device'})
diagram = ET.SubElement(mxfile, 'diagram', {'id':'diffusion-overview', 'name':'Diffusion overview'})
model = ET.SubElement(diagram, 'mxGraphModel', {'dx':str(W), 'dy':str(H), 'grid':'1', 'gridSize':'10', 'guides':'1', 'tooltips':'1', 'connect':'1', 'arrows':'1', 'fold':'1', 'page':'1', 'pageScale':'1', 'pageWidth':str(W), 'pageHeight':str(H), 'math':'0', 'shadow':'0', 'background':'#ffffff'})
root = ET.SubElement(model, 'root')
ET.SubElement(root, 'mxCell', {'id':'0'})
ET.SubElement(root, 'mxCell', {'id':'1', 'parent':'0'})
count = 0

def uid(prefix):
    global count
    count += 1
    return f'{prefix}-{count}'

def cell_vertex(name, x, y, w, h, style, value='', parent='1'):
    c = ET.SubElement(root, 'mxCell', {'id':name, 'value':value, 'style':style, 'vertex':'1', 'parent':parent})
    ET.SubElement(c, 'mxGeometry', {'x':str(x), 'y':str(y), 'width':str(w), 'height':str(h), 'as':'geometry'})
    return name

def group(name, x, y, w, h):
    cell_vertex(name, x, y, w, h, 'group;container=1;collapsible=0;recursiveResize=0;')
    return (name, x, y)

def local(x, y, parent):
    return (x-parent[1], y-parent[2]) if parent else (x,y)

def rect(x,y,w,h,fill='white',stroke=INK,sw=1.6,parent=None):
    xx,yy = local(x,y,parent)
    cell_vertex(uid('box'), xx,yy,w,h, f'rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={sw};', parent=parent[0] if parent else '1')
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    draw.rectangle((x*SCALE,y*SCALE,(x+w)*SCALE,(y+h)*SCALE), fill=fill, outline=stroke, width=max(1,round(sw*SCALE)))

def dots(cx,cy):
    # Native circles avoid reliance on a mathematical ellipsis font glyph.
    for dx in (-12,0,12):
        x,y,r=cx+dx,cy,2.5
        cell_vertex(uid('dot'),x-r,y-r,2*r,2*r,f'ellipse;html=1;fillColor={GRAY};strokeColor=none;')
        svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{GRAY}"/>')
        draw.ellipse(((x-r)*SCALE,(y-r)*SCALE,(x+r)*SCALE,(y+r)*SCALE),fill=GRAY)

def line(points,color=INK,sw=2,arrow=False,dashed=False,parent=None):
    style=f'edgeStyle=none;noEdgeStyle=1;rounded=0;html=1;strokeColor={color};strokeWidth={sw};startArrow=none;endArrow={"block" if arrow else "none"};endFill=1;endSize=12;'
    if dashed: style += 'dashed=1;dashPattern=4 4;'
    c=ET.SubElement(root,'mxCell', {'id':uid('line'),'value':'','style':style,'edge':'1','parent':parent[0] if parent else '1'})
    g=ET.SubElement(c,'mxGeometry', {'relative':'1','as':'geometry'})
    lp=[local(x,y,parent) for x,y in points]
    for name,pt in [('sourcePoint',lp[0]),('targetPoint',lp[-1])]:
        ET.SubElement(g,'mxPoint',{'x':str(round(pt[0],3)),'y':str(round(pt[1],3)),'as':name})
    if len(lp)>2:
        a=ET.SubElement(g,'Array',{'as':'points'})
        for xx,yy in lp[1:-1]: ET.SubElement(a,'mxPoint',{'x':str(round(xx,3)),'y':str(round(yy,3))})
    pts=' '.join(f'{x:.3f},{y:.3f}' for x,y in points)
    dash=' stroke-dasharray="8 8"' if dashed else ''
    svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"{dash}/>')
    if dashed:
        for (x1,y1),(x2,y2) in zip(points, points[1:]):
            dist=math.hypot(x2-x1,y2-y1)
            for t in range(0,math.ceil(dist),16):
                a,b=t/dist,min(t+8,dist)/dist
                draw.line(((x1+(x2-x1)*a)*SCALE,(y1+(y2-y1)*a)*SCALE,(x1+(x2-x1)*b)*SCALE,(y1+(y2-y1)*b)*SCALE),fill=color,width=round(sw*SCALE))
    else:
        draw.line([(x*SCALE,y*SCALE) for x,y in points],fill=color,width=max(1,round(sw*SCALE)),joint='curve')
    if arrow:
        x,y=points[-1]; px,py=points[-2]
        angle=math.atan2(y-py,x-px)
        length,half=14,6
        arr=[(x,y),(x-length*math.cos(angle)+half*math.sin(angle),y-length*math.sin(angle)-half*math.cos(angle)),(x-length*math.cos(angle)-half*math.sin(angle),y-length*math.sin(angle)+half*math.cos(angle))]
        svg.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in arr)+f'" fill="{color}"/>')
        draw.polygon([(a*SCALE,b*SCALE) for a,b in arr],fill=color)

def font(size,bold=False,italic=False):
    filename='arialbi.ttf' if bold and italic else 'arialbd.ttf' if bold else 'ariali.ttf' if italic else 'arial.ttf'
    return ImageFont.truetype('C:/Windows/Fonts/'+filename,round(size*SCALE))

def text(value,x,y,w,h,size=25,color=INK,bold=False,parent=None):
    xx,yy=local(x,y,parent)
    cell_vertex(uid('text'),xx,yy,w,h,f'text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;overflow=visible;spacing=0;fontFamily=Arial;fontSize={size};fontColor={color};fontStyle={1 if bold else 0};',escape(value),parent[0] if parent else '1')
    svg.append(f'<text x="{x+w/2}" y="{y+h/2}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{"700" if bold else "400"}" fill="{color}" text-anchor="middle" dominant-baseline="central">{escape(value)}</text>')
    draw.text(((x+w/2)*SCALE,(y+h/2)*SCALE),value,font=font(size,bold),fill=color,anchor='mm')

def math_text(parts,x,y,w,h,size=31,color=INK,parent=None):
    # parts are (text, is_subscript, italic). All math remains editable text.
    markup=''.join((f'<sub>{escape(v)}</sub>' if sub else f'<i>{escape(v)}</i>' if it else escape(v)) for v,sub,it in parts)
    xx,yy=local(x,y,parent)
    cell_vertex(uid('math'),xx,yy,w,h,f'text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;spacing=0;fontFamily=Arial;fontSize={size};fontColor={color};',markup,parent[0] if parent else '1')
    widths=[]
    for v,sub,it in parts:
        f=font(size*.70 if sub else size,italic=it)
        widths.append(draw.textlength(v,font=f)/SCALE)
    left=x+(w-sum(widths))/2
    for (v,sub,it),wid in zip(parts,widths):
        fs=size*.70 if sub else size
        baseline=y+h/2+size*.34+(size*.22 if sub else 0)
        svg.append(f'<text x="{left}" y="{baseline}" font-family="Arial, Helvetica, sans-serif" font-size="{fs}" font-style="{"italic" if it else "normal"}" fill="{color}">{escape(v)}</text>')
        draw.text((left*SCALE,baseline*SCALE),v,font=font(fs,italic=it),fill=color,anchor='ls')
        left+=wid

# Direction labels and arrows: color is redundant with arrow direction and text.
text('Forward diffusion',0,12,W,40,30,BLUE,True)
text('Add Gaussian noise',0,52,W,32,23,GRAY)
line([(188,108),(1312,108)],BLUE,3,True)

rng=random.Random(20260917)
n=88
signals=[]
for ch in range(3):
    clean=[.60*math.sin(2*math.pi*(2.1*i/(n-1)+ch*.16))+.18*math.sin(2*math.pi*4.7*i/(n-1)+ch) for i in range(n)]
    # Schematic realizations of a progressive Gaussian-noising chain.
    levels=[clean]
    for retain in (.84,.82,.001):
        levels.append([math.sqrt(retain)*v+math.sqrt(1-retain)*rng.gauss(0,.72) for v in levels[-1]])
    signals.append(levels)

xs=[58,432,808,1182]
subs=['0','s−1','s','S']
for stage,(x,sub) in enumerate(zip(xs,subs)):
    p=group(f'state-{stage}',x,147,260,250)
    math_text([('X',False,True),(sub,True,False)],x,146,260,40,31,parent=p)
    rect(x,194,260,156,'#FAFCFE',INK,1.7,parent=p)
    for ch in range(3):
        cy=225+ch*47
        line([(x+15,cy),(x+245,cy)],GRID,1,parent=p)
        values=signals[ch][stage]
        pts=[(x+16+228*i/(n-1),cy-max(-1.7,min(1.7,v))*11.7) for i,v in enumerate(values)]
        line(pts,BLUE if stage<3 else '#657380',1.85,parent=p)
    if stage==0: text('Data',x,359,260,32,24,parent=p)
    elif stage==3: text('Gaussian noise',x,359,260,32,24,parent=p)

# Ellipses denote omitted diffusion steps, not a different process.
dots(375,272)
line([(708,253),(790,253)],BLUE,2.6,True)
line([(790,295),(708,295)],ORANGE,2.6,True)
dots(1125,272)

line([(1312,426),(188,426)],ORANGE,3,True)
text('Reverse denoising',70,456,440,44,30,ORANGE,True)
rect(563,458,530,62,'#FFF9F3',ORANGE,1.7)
text('Noise predictor',570,470,244,36,23,INK)
math_text([('ε',False,False),('θ',True,False),('(',False,False),('X',False,True),('s',True,False),(', s)',False,False)],821,470,253,36,28)
line([(828,458),(828,426)],ORANGE,1.7,True,True)

svg.append('</svg>')
stem=OUT/'diffusion_overview'
ET.indent(mxfile,space='  ')
ET.ElementTree(mxfile).write(stem.with_suffix('.drawio'),encoding='utf-8',xml_declaration=True)
stem.with_suffix('.svg').write_text('\n'.join(svg),encoding='utf-8')
im.save(stem.with_suffix('.png'),dpi=(300,300))
notes='''# Diffusion figure

- diffusion_overview.drawio: editable native rectangles, labels, arrows, and waveform polylines; no embedded raster images.
- diffusion_overview.svg: vector preview for publication.
- diffusion_overview.png: 4500 × 1650 px preview at 300 dpi.

Open the .drawio file in diagrams.net. Each state is grouped; ungroup to edit individual waveforms and labels. Text and connectors elsewhere can be edited directly.

The diffusion-step notation s and S matches section_3_1_preliminaries.md, Section 3.1.2. The waveforms are schematic, not experimental results. The two directions show a forward Gaussian-noising chain and learned reverse transitions. Reverse generation does not guarantee recovery of the exact original sample. The terminal noise state illustrates the usual sufficiently-noised DDPM limit; it is not a claim about a particular finite schedule in the FMD implementation. The generic noise predictor is architecture-neutral.

Suggested caption: Forward diffusion and learned reverse denoising for time-series data. Gaussian noise is progressively added to the data, while a learned noise predictor guides the reverse process.

Reference: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. https://arxiv.org/abs/2006.11239
'''
(OUT/'diffusion_overview_notes.md').write_text(notes,encoding='utf-8')

# Structural checks: valid XML, unique cells, valid parents, and no raster assets.
tree=ET.parse(stem.with_suffix('.drawio'))
cells=tree.findall('.//mxCell')
ids=[c.attrib['id'] for c in cells]
assert len(ids)==len(set(ids))
assert all(c.attrib.get('parent','0') in ids for c in cells)
assert not any('image=' in c.attrib.get('style','') for c in cells)
ET.parse(stem.with_suffix('.svg'))
assert im.size == (4500,1650)
print(f'Created {stem.name}: .drawio, .svg, .png; {len(cells)} editable cells; XML checks passed.')
