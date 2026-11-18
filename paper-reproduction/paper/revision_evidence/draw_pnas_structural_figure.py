from pathlib import Path
from html import escape
O=Path(__file__).resolve().parents[1]/'figures'
S=[]; INK='#202326'; LINE='#70777b'; BLUE='#a9c9e2'; ORANGE='#f39a28'
def put(s):S.append(s)
def t(x,y,s,z=14,c=INK,a='middle',bold=False):
 put(f'<text x="{x}" y="{y}" fill="{c}" font-size="{z}" text-anchor="{a}"'+(' font-weight="bold"' if bold else '')+f'>{escape(s)}</text>')
def p(d,c=LINE,w=.9,dash=None,arr=False):
 put(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{w}" stroke-linecap="round" stroke-linejoin="round"'+(f' stroke-dasharray="{dash}"' if dash else '')+(' marker-end="url(#a)"' if arr else '')+'/>')
def node(x,y,r=5,fill='white',stroke=LINE,w=.8):
 put(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{w}"/>')
def sq(x,y,w,h,fill='white',stroke='none'):
 put(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width=".7"/>')
def mat(x,y,rows=5,cols=6,s=6,new=False):
 for i in range(rows):
  for j in range(cols):sq(x+j*s,y+i*s,s-1,s-1,ORANGE if new and i==rows-1 else ['#edf3f8','#cfdfec',BLUE,'#fafafa'][(i*3+j*5+i*j)%4])
 p(f'M{x-4} {y-2} h-4 v{rows*s+3} h4 M{x+cols*s+3} {y-2} h4 v{rows*s+3} h-4',INK,.8)
def fp(x,y,scale=1,changed=False,labels=False):
 ns=[(0,0),(-37,37),(34,37),(-58,74),(-13,74),(34,74),(-58,111),(-13,111),(34,111),(70,111)]
 es=[(0,1),(0,2),(1,3),(1,4),(2,5),(3,6),(4,7),(5,8),(5,9)]
 for a,b in es:
  ax,ay=ns[a];bx,by=ns[b];p(f'M{x+ax*scale} {y+ay*scale} L{x+bx*scale} {y+by*scale}',ORANGE if changed and b==9 else LINE,.85)
 for i,(xx,yy) in enumerate(ns):node(x+xx*scale,y+yy*scale,5*scale,BLUE if i else 'white',ORANGE if changed and i in [0,2,5,9] else LINE,1.6 if changed and i in [0,2,5,9] else .7)
 if labels:
  t(x,y-11,'∅',13)
  for i,label in [(1,'x₁ = 0'),(2,'x₁ = 1'),(3,'x₂ = 0'),(4,'x₂ = 1'),(5,'x₂ = 1')]:
   xx,yy=ns[i];t(x+xx*scale,y+yy*scale-11,label,11)
def rules(x,y,kind='A'):
 # Three representative rule terms. No invented measured rule coefficients.
 for j in range(3):
  yy=y+j*26
  t(x-63,yy+4,'r'+['₁','₂','K'][j],13)
  for k in range(3):node(x-35+k*19,yy,4,BLUE,LINE)
  p(f'M{x+9} {yy} H{x+28}',LINE,.7)
  t(x+48,yy+5,'φ'+['₁','₂','K'][j],16)
 t(x,y+86,'rule-response basis',11,LINE)
def bracket(x1,x2,y,label):
 p(f'M{x1} {y+7} V{y} H{x2} V{y+7}',LINE,.75);t((x1+x2)/2,y-8,label,12)

put('<svg xmlns="http://www.w3.org/2000/svg" width="972pt" height="489.6pt" viewBox="0 0 1350 680"><rect width="1350" height="680" fill="white"/><defs><marker id="a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#70777b"/></marker></defs><g font-family="Arial,sans-serif">')
t(23,29,'a',18,a='start',bold=True);t(48,29,'Shared pattern representation',15,a='start',bold=True)
t(461,29,'b',18,a='start',bold=True);t(486,29,'Complementary scoring on the same query x',15,a='start',bold=True)
p('M433 45 V388','#d0d0d0',.65)
# Open structural drawing: objects, not enclosing process rectangles.
mat(42,103,6,6,7);t(62,88,'Dₜ',18);t(62,164,'revealed records',11,LINE)
p('M99 124 H125',LINE,.8,arr=True)
fp(209,95,1.15,False,True)
t(210,254,'FP tree  Tₜ',15);t(210,275,'class counts at every node',11,LINE)
p('M298 177 H325',LINE,.8,arr=True)
t(368,102,'Pₜ',18)
for y,s in [(139,'{x₁ = 0}'),(170,'{x₁ = 0,'),(186,'  x₂ = 1}'),(220,'···')]:t(368,y,s,12)
t(224,328,'support ≥ 0.1     length ≤ 3',12)
t(224,351,'Fixed item order; only new records are inserted',11,LINE)
# Shared provenance supplied to two selectors; small arrow, not a workflow rail.
p('M388 239 H422 V67 H550 V83',LINE,.8,arr=True)
p('M422 67 H989 V83',LINE,.8,arr=True)
t(566,108,'Joint projection',13);t(566,128,'32 rules / class',11,LINE)
bracket(806,1215,118,'Coverage-complementary selection · one shared 64-rule set')
for x in [838,1165]:p(f'M{x} 127 V141',LINE,.7,arr=True)

t(566,168,'A',14,bold=True);t(838,168,'B',14,bold=True);t(1165,168,'C',14,bold=True)
rules(566,192)
# HVDM: categorical feature distances between query and a rule reference.
for j in range(4):
 node(791+j*22,204,4.5,BLUE,LINE)
 node(791+j*22,239,4.5,'white',LINE)
 p(f'M{791+j*22} 211 V232',LINE,.65,dash='2 2')
t(778,208,'x',13,a='end');t(778,243,'r',13,a='end')
t(838,278,'HVDM rule distance',11,LINE)
# A scored historical case table with an explicit score column.
mat(1106,191,5,6,7)
p('M1157 188 V227',LINE,.6)
for j,s in enumerate(['−','+','−','+','+']):t(1170,197+j*7,s,9)
t(1146,250,'labeled score history',11,LINE)
# Selective review is shown as a sparse change to a score vector.
for j in range(5):node(1118+j*20,274,4,BLUE,ORANGE if j==3 else LINE,1.4 if j==3 else .7)
t(1160,296,'conditional review + stability',11,LINE)

# The arithmetic layout deliberately follows the tree-sum figure's plus-separated objects.
t(566,333,'sA(x)',22);t(838,333,'sB(x)',22);t(1165,333,'sC(x)',22)
t(697,332,'+',26);t(1006,332,'+',26)
p('M490 347 H1236',INK,1);t(855,373,'3',21)
t(566,399,'Sugeno rule average',12)
t(838,399,'ℓ(x) + β h(x) / 10',15)
t(1165,399,'reviewed coverage Sugeno',12)
t(855,429,'s(x) ≥ 0 : malignant',16)

# Time is represented by successive states, as in the reference figure's iterations.
p('M23 461 H1327','#c8c8c8',.75)
t(23,489,'c',18,a='start',bold=True);t(48,489,'Prediction precedes label release and update',15,a='start',bold=True)
t(951,489,'d',18,a='start',bold=True);t(976,489,'Fixed holdout',15,a='start',bold=True)
p('M925 480 V655','#d0d0d0',.65)
for x in [212,420,630]:p(f'M{x} 517 V633','#d0d0d0',.6)
for x,s in [(112,'Predict batch'),(316,'Reveal labels'),(526,'Update tree'),(773,'Refit for next batch')]:t(x,521,s,12)
mat(94,545,5,5,6);t(112,607,'scores saved',11,LINE)
mat(292,545,5,5,6)
for j in range(5):node(333,548+j*6,2.1,BLUE,ORANGE,1)
t(316,607,'new labeled records',11,LINE)
fp(520,537,.45,True);t(526,620,'insert → remine',11,LINE)
fp(716,537,.45,True)
p('M755 568 H783',LINE,.8,arr=True);t(809,558,'sA',14);t(809,577,'sB',14);t(809,596,'sC',14)
t(773,633,'cumulative prefix',11,LINE)
t(458,665,'Scaling frozen at 300 records; the updated model is used only for subsequent predictions',11,LINE)

# All checkpoint evaluations target one unchanged sample set within the split.
for x,n in zip([981,1083,1185,1287],[300,500,700,849]):
 t(x,529,str(n),11,LINE);node(x,549,5,BLUE,LINE)
 p(f'M{x} 555 V581',LINE,.7)
p('M981 581 H1287 M1134 581 V598',LINE,.8,arr=True)
mat(1106,607,3,8,6)
t(1134,651,'same 213 records',12)
t(1134,669,'excluded from fitting',11,LINE)
put('</g></svg>')
(O/'fig01_three_way_fusion_en_pnas_structure.svg').write_text(''.join(S),encoding='utf-8')
print('SVG written')
