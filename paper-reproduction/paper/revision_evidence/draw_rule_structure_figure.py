from pathlib import Path
from html import escape
import math

OUT = Path(__file__).resolve().parents[1] / 'figures'
INK='#30353b'; LINE='#707980'; BLUE='#496b8a'; RUST='#a66a40'; GREEN='#718362'
P=[]
def add(s): P.append(s)
def text(x,y,s,size=14,color=INK,anchor='middle',weight='normal',italic=False):
    add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}"'+(' font-style="italic"' if italic else '')+f'>{escape(s)}</text>')
def path(d,color=LINE,w=1.2,fill='none',dash=None,arrow=False):
    add(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{w}" stroke-linejoin="round"'+(f' stroke-dasharray="{dash}"' if dash else '')+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
def rect(x,y,w,h,fill='white',stroke=LINE,r=0):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="0.9"/>')
def circ(x,y,r=7,fill='white',stroke=LINE):
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
def hollow(x,y,w=33):
    path(f'M{x} {y-4} h{w-10} v-5 l10 9 -10 9 v-5 H{x} Z',LINE,.9,'white')
def matrix(x,y,cols=8,rows=9,cell=7,color=BLUE):
    colors=['#f5f7f9','#e1e8ef','#bdcddb',color]
    for i in range(rows):
        for j in range(cols):
            rect(x+j*cell,y+i*cell,cell-1,cell-1,colors[(i*7+j*3+i*j)%4],'none')
    path(f'M{x-4} {y-2} h-4 v{rows*cell+2} h4 M{x+cols*cell+3} {y-2} h4 v{rows*cell+2} h-4',INK,1)
def rule_strip(x,y,n=5,color=BLUE,scale=1):
    w=110*scale;h=23*scale
    rect(x,y,w,h,'white',LINE,h/2)
    for i in range(n):
        circ(x+(13+i*21)*scale,y+h/2,6.7*scale,color,'none')
    text(x+w/2,y+h+15*scale,'r₁   r₂   ···   rK',11*scale,color)
def tree(x,y,s=1):
    nodes=[(0,0),( -37,36),(6,36),(48,36),(-56,72),(-23,72),(5,72),(43,72),(-58,108),(-20,108),(24,108),(57,108)]
    edges=[(0,1),(0,2),(0,3),(1,4),(1,5),(2,6),(3,7),(4,8),(5,9),(6,10),(7,11)]
    for a,b in edges:
        ax,ay=nodes[a];bx,by=nodes[b];path(f'M{x+ax*s} {y+ay*s} L{x+bx*s} {y+by*s}',LINE,1)
    for i,(xx,yy) in enumerate(nodes):
        circ(x+xx*s,y+yy*s,7*s,'#dde6ed' if i else 'white',BLUE)
    text(x,y+4*s,'∅',10*s)
def inference(x,y,color):
    # Explicit rule activation / consequent / summation structure, not layer icons.
    for j in range(3):
        yy=y+j*30
        circ(x,yy,10,'white',color);text(x,yy+4,'μ',13,color)
        path(f'M{x+10} {yy} H{x+39}',color,1)
        circ(x+50,yy,10,'white',color);text(x+50,yy+4,'f',13,color,italic=True)
        path(f'M{x+60} {yy} L{x+101} {y+30}',color,1)
    circ(x+113,y+30,12,'white',color);text(x+113,y+35,'Σ',17,color)
    if y < 300: text(x+25,y+88,'rule responses',11,color)
def score(x,y,label,color):
    rect(x,y-16,49,32,'white',color,16);text(x+24.5,y+5,label,17,color,italic=True)
def mini_sheet(x,y,color=BLUE,labels=False):
    rect(x,y,30,36,'white',LINE)
    for i in range(4):
        path(f'M{x+5} {y+7+i*7} H{x+16}',LINE,.8)
        if labels:circ(x+23,y+7+i*7,2.2,color,'none')
        else:path(f'M{x+21} {y+8+i*7} l2 -3 2 3',color,1)

add('<svg xmlns="http://www.w3.org/2000/svg" width="936pt" height="540pt" viewBox="0 0 1248 720">')
add('<rect width="1248" height="720" fill="white"/>')
add(f'<defs><marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{LINE}"/></marker></defs>')
add('<g font-family="Arial, sans-serif">')
text(22,26,'a',19,weight='bold',anchor='start')
# The stage arrangement and vector capsules follow MVRL_FS Fig. 2.
for x,w,title in [(22,213,'Shared pattern space'),(240,268,'Rule representation'),(513,498,'Three complementary scores'),(1016,210,'Decision')]:
    rect(x,43,w,430,'white','#98a0a6')
    text(x+w/2,65,title,15,weight='bold')
    path(f'M{x} 77 H{x+w}','#bcc2c6',.8)

text(127,105,'Revealed prefix  Dₜ',15,italic=True)
matrix(94,124,9,7,7)
text(127,190,'Nₜ × 18',12)
hollow(110,213,33)
tree(128,239,.9)
text(127,365,'Class-counted FP tree',14)
text(127,386,'fixed item order · insert only',11)
text(127,432,'support ≥ 0.1  |  length ≤ 3',12)
path('M186 302 H249 V142 H330',LINE,1.1,arrow=True)

# Candidate incidence representation: two classes are delineated structurally.
text(375,105,'Candidate patterns  Pₜ',15,italic=True)
matrix(341,120,8,6,7)
text(375,178,'pattern–item incidence',11)
path('M375 186 V205 H303 V221',LINE,1.2,arrow=True)
path('M375 205 H434 V294',LINE,1.2,arrow=True)
text(317,239,'Joint projection',13,BLUE)
rule_strip(262,252,color=BLUE)
text(317,307,'32 / class',12,BLUE)
text(428,316,'Coverage selection',13,RUST)
rule_strip(375,330,color=RUST)
text(430,385,'one shared set · 64 rules',12,RUST)
path('M375 264 H494 V162 H528',BLUE,1.3,arrow=True)
path('M488 342 H502 V282 H528',RUST,1.3,arrow=True)
path('M502 342 V410 H528',RUST,1.3,arrow=True)

# A: native fuzzy inference structure.
text(531,105,'A   Projection-selected Sugeno',14,BLUE,anchor='start',weight='bold')
inference(578,132,BLUE)
text(763,150,'Σ wᵣ(x) fᵣ(x)',17,BLUE,italic=True)
path('M750 160 H875',BLUE,.8)
text(812,179,'Σ wᵣ(x)',17,BLUE,italic=True)
path('M703 162 H730',BLUE,1,arrow=True)
path('M895 162 H928',BLUE,1,arrow=True)
score(938,162,'sA',BLUE)

# B: an additive calibrated correction, never a probability average.
path('M513 226 H1011','#d5d9dc',.8)
text(531,251,'B   Logistic + HVDM correction',14,RUST,anchor='start',weight='bold')
text(607,293,'ℓ(x)',23,RUST,italic=True)
text(607,314,'cumulative logistic',11,RUST)
text(783,293,'β h(x) / 10',21,RUST,italic=True)
text(783,314,'calibrated distance score',11,RUST)
text(701,293,'+',24,RUST)
path('M883 285 H935',RUST,1,arrow=True)
score(938,285,'sB',RUST)

# C: a score corrected conditionally by revealed-history checks.
path('M513 348 H1011','#d5d9dc',.8)
text(531,374,'C   Coverage Sugeno + conditional repair',14,GREEN,anchor='start',weight='bold')
inference(552,399,GREEN)
path('M677 429 H698',GREEN,1,arrow=True)
# Historical evidence as a true sample-neighbour relation rather than a process box.
for xx,yy in [(733,401),(757,405),(721,424),(759,433),(738,449)]:
    path(f'M740 423 L{xx} {yy}',GREEN,.75)
    circ(xx,yy,3.5,'#dce5d5',GREEN)
circ(740,423,5,'white',GREEN)
text(740,464,'history',10,GREEN)
path('M771 423 H789',GREEN,1,arrow=True)
for i,lab in enumerate(['R','F','I']):
    circ(811+i*34,423,11,'white',GREEN);text(811+i*34,427,lab,12,GREEN)
    if i<2:path(f'M{823+i*34} 423 H{833+i*34}',GREEN,.9)
path('M892 423 H935',GREEN,1,arrow=True)
score(938,423,'sC',GREEN)

# Scores converge once and only once.
for yy in [162,285,423]:
    path(f'M988 {yy} H1037 L1091 285',LINE,1.2)
circ(1107,285,17,'white',INK);text(1107,291,'Σ',23)
text(1139,290,'/ 3',17,anchor='start')
path('M1107 303 V350',LINE,1.2,arrow=True)
text(1107,375,'s(x) ≥ 0',20,italic=True)
text(1107,399,'malignant',13)
text(1107,438,'equal-weight raw scores',11)

# Protocol panels are deliberately objects and timing, not prose-filled modules.
text(22,516,'b',19,weight='bold',anchor='start')
text(46,516,'Predict first, then update',15,weight='bold',anchor='start')
path('M22 529 H807','#98a0a6',.8)
xs=[71,225,379,533,687]
labs=['Current model','Batch predictions','Reveal labels','Insert + remine','Refit next model']
for x,lab in zip(xs,labs):text(x,627,lab,12)
rule_strip(37,566,5,BLUE,.62)
mini_sheet(210,557,BLUE,False)
mini_sheet(364,557,RUST,True)
tree(534,548,.43)
rule_strip(653,566,5,GREEN,.62)
for x in [111,265,419,573]:hollow(x,578,70)
path('M298 542 V607',RUST,1.2,dash='3 3')
text(688,659,'used for the next batch',11,GREEN)
path('M687 642 V674 H71 V600',LINE,1,dash='4 3',arrow=True)
text(379,696,'Initial prefix: 300   |   frozen scaling   |   cumulative refitting',11)

text(841,516,'c',19,weight='bold',anchor='start')
text(866,516,'Fixed holdout within each split',15,weight='bold',anchor='start')
path('M841 529 H1226','#98a0a6',.8)
text(1035,554,'Training prefix',12)
for x,n in zip([882,977,1072,1167],[300,500,700,849]):
    circ(x,585,7,'#e1e8ef',BLUE)
    text(x,573,str(n),12)
    path(f'M{x} 594 V621',LINE,.9)
path('M882 621 H1167 M1024 621 V642',LINE,1,arrow=True)
matrix(989,651,9,3,6)
text(1035,696,'same 213 records · excluded from fitting',11)
add('</g></svg>')
f=OUT/'fig01_three_way_fusion_en_rule_structure.svg'
f.write_text(''.join(P),encoding='utf-8')
print(f)
