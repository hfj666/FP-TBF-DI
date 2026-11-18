"""Vector method overview, structurally referenced to MFMR-FN Fig. 1."""
from pathlib import Path
from html import escape
import argparse
import re
import math

OUT=Path(__file__).resolve().parents[1]/'figures'
S=[]
parser = argparse.ArgumentParser()
parser.add_argument('--refined', action='store_true')
parser.add_argument('--reviewed', action='store_true')
parser.add_argument('--dependencies', action='store_true')
parser.add_argument('--publication', action='store_true')
parser.add_argument('--history', action='store_true')
parser.add_argument('--compact', action='store_true')
parser.add_argument('--feedback', action='store_true')
parser.add_argument('--online-loop', action='store_true')
parser.add_argument('--scoring-clarified', action='store_true')
options = parser.parse_args()
scoring_clarified = options.scoring_clarified
online_loop = options.online_loop or scoring_clarified
feedback = options.feedback or online_loop
compact = options.compact or feedback
history = options.history or compact
publication = options.publication or history
dependencies = options.dependencies or publication
reviewed = options.reviewed or dependencies
refined = options.refined or reviewed

def format_label(label):
 if not refined:
  return escape(label)
 parts = re.split(r'(sA|sB|sC|sbase|wA|cA)', label)
 subscripts = {'sA': ('s','A'), 'sB': ('s','B'), 'sC': ('s','C'),
               'sbase': ('s','base'), 'wA': ('w','A'), 'cA': ('c','A')}
 return ''.join(
  f'{subscripts[part][0]}<tspan baseline-shift="sub" font-size="72%">{subscripts[part][1]}</tspan>'
  if part in subscripts else escape(part) for part in parts
 )
INK='#332d33'; EDGE='#786b73'; LIGHT='#e6e0e4'
QUERY_INK='#325e83'
PINK='#c898b8'; MAUVE='#b888a8'; LILAC='#c8a8c8'; SAND='#d8b8a8'
TEAL='#cae1db'; BLUE='#9bb3cc'; GREEN='#b8c8b2'
def out(s):S.append(s)
def txt(x,y,s,size=16,anchor='middle',color=INK,bold=False,rotate=None):
 if reviewed:
  size=max(15,size)
  if color==EDGE: color='#574e56'
 out(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}"'+(' font-weight="bold"' if bold else '')+(f' transform="rotate({rotate} {x} {y})"' if rotate is not None else '')+f'>{format_label(s)}</text>')
def path(d,color=INK,width=1.1,arrow=False,dash=None):
 out(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"'+(' marker-end="url(#tip)"' if arrow else '')+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
def rect(x,y,w,h,fill='white',stroke=EDGE,width=.8,dash=None):
 out(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
def circle(x,y,r=6,fill='white',stroke=EDGE,width=1):
 out(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')
def polygon(points,fill='white',stroke=EDGE,width=.8):
 out('<polygon points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')
def plane(x,y,cols=7,rows=6,cell=10,kind='incidence',layers=1,palette=None):
 colors=palette or [PINK,MAUVE,LILAC,SAND,'#d8a898','#c4bdca']
 for layer in range(layers-1,-1,-1):
  xx=x+layer*7;yy=y-layer*5
  out(f'<g transform="matrix(1 -.22 0 1 {xx} {yy})">')
  rect(0,0,cols*cell,rows*cell,'#faf7f9',EDGE,.6)
  for r in range(rows):
   for c in range(cols):
    if kind=='incidence':fill=colors[(r+c)%len(colors)] if (r*5+c*3)%7<2 else '#f6f0f4'
    elif kind=='history':fill=colors[(r*2+c*3+r*c)%len(colors)]
    elif kind=='mask':fill=colors[0] if (r+c*2)%3==1 else '#f5f1f4'
    else:fill=colors[(r*2+c*3+r*c)%len(colors)]
    rect(c*cell,r*cell,cell,cell,fill,'#ffffff',.35)
  out('</g>')
def bar(x,y,w,h,label,fill):
 rect(x,y,w,h,fill,'none');txt(x+w/2+5,y+h/2,label,15,rotate=-90)
def arr(x1,y1,x2,y2):path(f'M{x1} {y1} H{x2}' if y1==y2 else f'M{x1} {y1} L{x2} {y2}',arrow=True)
def query_port(x,y,end_x,end_y):
 txt(x,y+5,'x ∈ Q' if publication else 'x ∈ Qₜ',16,anchor='end',color=QUERY_INK)
 path(f'M{x+6} {y} H{end_x-8} V{end_y} H{end_x}',QUERY_INK,1.5)
 polygon([(end_x,end_y),(end_x-6,end_y-3),(end_x-6,end_y+3)],QUERY_INK,QUERY_INK)
def tree(x,y,scale=1,highlight=False):
 nodes=[(0,0),(-63,40),(22,40),(83,40),(-82,84),(-38,84),(23,84),(71,84),(-82,128),(-38,128),(23,128),(71,128),(106,128)]
 edges=[(0,1),(0,2),(0,3),(1,4),(1,5),(2,6),(3,7),(4,8),(5,9),(6,10),(7,11),(7,12)]
 for a,b in edges:
  ax,ay=nodes[a];bx,by=nodes[b];path(f'M{x+ax*scale} {y+ay*scale} L{x+bx*scale} {y+by*scale}',MAUVE if highlight and b in [3,7,12] else EDGE,1)
 for i,(xx,yy) in enumerate(nodes):circle(x+xx*scale,y+yy*scale,6*scale,'white' if i==0 else '#e9dce6',MAUVE if highlight and i in [0,3,7,12] else EDGE,1.5 if highlight and i in [0,3,7,12] else .8)
 if scale>.7:
  txt(x,y-12,'∅',14)
  for i,label in [(1,'x₁=0'),(2,'x₁=1'),(3,'x₁=2'),(4,'x₂=0'),(5,'x₂=1'),(6,'x₂=1')]:
   xx,yy=nodes[i];txt(x+xx*scale,y+yy*scale-11,label,12)
  txt(x+106*scale+18,y+128*scale+3,'···',14)
def gaussian(x,y,w=80,h=38):
 path(f'M{x} {y} H{x+w}',EDGE,.7)
 for center,spread,col in [(w*.28,10,MAUVE),(w*.5,13,BLUE),(w*.73,9,SAND)]:
  ps=[]
  for i in range(45):
   u=w*i/44;v=y-h*math.exp(-.5*((u-center)/spread)**2);ps.append((x+u,v))
  path('M'+' L'.join(f'{a:.2f} {b:.2f}' for a,b in ps),col,1.6)
def scores(x,y,color):
 for i,c in enumerate([color,'#ece5eb',color,'#e7e2e6',color]):rect(x,y+i*11,10,11,c,'white',.4)
def tiny_vector(x,y,colors,cell=8):
 for i,c in enumerate(colors):rect(x+i*cell,y,cell,cell,c,'white',.3)

canvas_height = 1012 if compact else 1030 if history else 1011 if publication else 1055 if reviewed else 950
out(f'<svg xmlns="http://www.w3.org/2000/svg" width="1036.8pt" height="{canvas_height*.72:g}pt" viewBox="0 0 1440 {canvas_height}"><rect width="1440" height="{canvas_height}" fill="white"/><defs><marker id="tip" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="4.5" markerHeight="4.5" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#332d33"/></marker></defs><g font-family="Arial,sans-serif">')

# The reference's two levels are retained as the dominant composition.
txt(35,28,'a  Shared FP-pattern construction',20,anchor='start',bold=True)
rect(30,42,1380,205 if publication else 249,'white',EDGE,.8,'10 6')
if publication: out('<g transform="translate(0 -25)">')
plane(101,128,8,7,10,'history',3)
txt(148,225,'Revealed prefix Dₜ',17)
txt(148,248 if publication else 252,'Nₜ × 18',14,color=EDGE)
arr(211,157,285,157)
# Ordered records are data, displayed as categorical items rather than process boxes.
for i,line in enumerate(['x₁=0   x₂=1   ···','x₁=2   x₂=0   ···','x₁=0   x₂=0   ···']):txt(367,134+i*25,line,15)
path('M300 108 h-8 v82 h8 M431 108 h8 v82 h-8',EDGE,.8)
txt(367,225,'Fixed item order',17)
arr(455,157,531,157)
tree(642,93,1)
txt(653,252,'Counted FP tree Tₜ',17)
txt(781,208,'n, n₊',13,color=EDGE)
if dependencies: txt(819,249,'Item class counts',17,color=EDGE)
path('M753 212 L776 200',EDGE,.65)
arr(789,157,889,157)
bar(904,112,27,90,'FP-growth',TEAL)
arr(947,157,1080,157)
plane(1110,134,9,7,10,'incidence',3)
txt(1164,238,'Candidate patterns Pₜ',17)
txt(1164,263,'support ≥ 0.1; length ≤ 3',14,color=EDGE)
if publication:
 out('</g>')
 out('<g transform="translate(0 -68)">' if compact else '<g transform="translate(0 -44)">')
# A short inter-panel transfer, as in the original figure. P_t is repeated below.
path('M720 322 V338' if compact else 'M720 301 V324',INK,3 if compact else 4,True)

txt(35,350,'b  Three scores for the shared query set Q' if publication else 'b  Three scores for the same query batch Q' if refined else 'b  Three complementary scoring branches',20,anchor='start',bold=True)
rect(30,365,1380,500 if reviewed else 431,'white',EDGE,.8,'10 6')

# Shared rule provenance. There is exactly one coverage selector and rule bank.
if reviewed:
 plane(86,432,6,4,9,'history',1,[BLUE,'#c4cedb','#edf1f4'])
 txt(116,496,'Query set Q' if publication else 'Query batch Qₜ',18 if dependencies else 17,color=QUERY_INK if dependencies else INK)
 txt(116,520,'Blue: query input' if dependencies else 'Shared by A, B, and C',16,color=QUERY_INK if dependencies else EDGE)
plane(86,582,8,7,9,'incidence',2)
txt(128,678,'Same Pₜ',16)
path('M182 603 H223 V451 H285',arrow=True)
path('M223 603 V646 H285',arrow=True)
bar(285,414,25,78,'Projection', '#e7d6e4')
plane(352,437,8,6,8,'incidence',2)
arr(311,451,341,451)
txt(391,520,'32 rules / class',14)
bar(285,601,25,91,'Coverage',TEAL)
plane(352,632,8,6,8,'incidence',2)
arr(311,646,341,646)
txt(392,709,'64 shared rules',14)
path('M433 648 H478 V584 H523',arrow=True)
path('M478 648 V755 H523' if reviewed else 'M478 648 V728 H523',arrow=True)
arr(433,451,523,451)

# A: zero-order Sugeno, showing actual Gaussian premises and constant consequents.
txt(557,398,'A  Projection-selected Sugeno',18,anchor='start',bold=True)
if dependencies: query_port(525,424,552,443)
elif reviewed: txt(535,426,'x ∈ Qₜ',15,anchor='end')
gaussian(546,461,85,42)
txt(588,494,'Normalized rule weights' if scoring_clarified else 'Normalized Gaussian firing' if reviewed else 'Gaussian firing',13,color=EDGE)
arr(642,451,690,451)
plane(709,436,1,5,9,'history',1,[BLUE,'#c4cedb','#e1dce6'])
txt(713,505,'wA(x)',15)
arr(733,451,833,451)
tiny_vector(795,420,[PINK,LILAC,SAND,MAUVE,PINK],8)
txt(812,411,'cA',14)
path('M835 424 H850 V436',arrow=True)
circle(850,451,13,'white',EDGE);txt(850,457,'⊙',20)
path('M865 451 H904',arrow=True)
circle(922,451,14,'white',EDGE);txt(922,457,'Σ',20)
txt(884,487,'weighted constants',13,color=EDGE)
arr(938,451,1025,451)
scores(1045,424,BLUE);txt(1050,410,'sA(Q)',16)

# B: the learned rule-distance score is calibrated and added to logistic raw score.
txt(557,540,'B  Logistic + HVDM correction',18,anchor='start',bold=True)
tiny_vector(542,568,[LILAC,PINK,SAND,LILAC,SAND,PINK],10)
tiny_vector(542,600,[SAND,LILAC,PINK,LILAC,PINK,SAND],10)
for i in range(6):path(f'M{547+10*i} 580 V596',EDGE,.6,dash='2 2')
if dependencies: query_port(511,573,539,573)
else: txt(532,576,'x ∈ Qₜ' if reviewed else 'x',14,anchor='end')
txt(532,609,'r',14,anchor='end')
bar(630,558,23,59,'HVDM',TEAL)
if dependencies:
 if history:
  path('M622 628 H641 V619',arrow=True)
  txt(559,634,'FP item counts',16)
 else:
  path('M641 633 V619',arrow=True)
  txt(616,652,'FP item class counts',16)
arr(607,584,629,584)
if reviewed:
 bar(672,552,24,70,'Sugeno',TEAL)
 arr(655,584,670,584)
 arr(698,584,712,584)
else:
 arr(667,584,708,584)
plane(718,570,1,5,9,'history',1,[PINK,LILAC,SAND])
txt(722,635,'h(Q)',15)
arr(742,584,784,584)
circle(810,584,18,'white',EDGE);txt(810,589,'×',20)
txt(811,618,'β / 10',15)
arr(832,584,904,584)
circle(922,584,14,'white',EDGE);txt(922,590,'+',22)
# ℓ(x) is not a sigmoid or a class probability.
txt(968,526,'ℓ(Q)',17)
txt(947,505,'Cumulative logistic',13,color=EDGE)
path('M968 535 V552 H922 V568',arrow=True)
arr(938,584,1025,584)
scores(1045,557,PINK);txt(1050,543,'sB(Q)',16)

# C: base coverage score, historical review, then stability-conditioned replacement.
if reviewed:
 txt(557,672,'C  Coverage Sugeno + historical review',18,anchor='start',bold=True)
 if dependencies: query_port(525,727,552,746)
 else: txt(535,729,'x ∈ Qₜ',15,anchor='end')
 gaussian(545,765,61,31)
 txt(577,800,'Sugeno base',15)
 arr(616,755,654,755)
 plane(665,737,4,5,7,'history',1)
 txt(679,800,'[Q, sbase]' if publication else '[Qₜ, sbase]',16 if publication else 15)
 arr(699,755,740,755)
 bar(745,727,26,60,'Review',TEAL)
 txt(749,831,'RBF-SVM, recurrence, full-prefix review',17 if publication else 15,color=EDGE)
 # Both labeled history and the current training prefix inform the review.
 plane(671,705,3,2,6,'history',1)
 txt(645,714,'Hₜ',16,anchor='end')
 txt(700,694,'History input when available' if history else 'Labeled post-initialization history' if publication else 'Past features, coverage scores, labels',17 if publication else 15,color=EDGE)
 path('M694 708 H752 V723',arrow=True)
 txt(788,714,'Dₜ',16)
 path('M786 719 H763 V724',arrow=True)
 if dependencies:
  path('M800 710 H960')
  path('M840 710 V729',arrow=True)
  path('M960 710 V729',arrow=True)
  arr(773,755,789,755)
  for x,line1,line2 in [(793,'Categorical','deletion'),(911,'Numeric pair','interactions')]:
   rect(x,732,102,48,'#eee5ed','none')
   txt(x+51,752,line1,17 if publication else 16)
   txt(x+51,771,line2,17 if publication else 16)
  arr(897,755,908,755)
  arr(1016,755,1034,755)
  if not history:
   txt(904,807,'Updates require eligibility and stability' if publication else 'Each step updates eligible scores only if stable',17 if publication else 16,color=EDGE)
 else:
  path('M800 710 H898 V725',arrow=True)
  path('M773 755 H818 V786 H961',arrow=True)
  path('M818 755 V744 H845',arrow=True)
  rect(851,728,101,32,'#eee5ed','none')
  txt(901,749,'Stable repairs',15)
  arr(954,744,961,744)
  polygon([(966,732),(966,797),(1000,755)],'#eee5ed',EDGE,.9)
  txt(902,779,'Reviewed score',15)
  txt(903,815,'Select repair if eligible and stable',15,color=EDGE)
  arr(1002,755,1034,755)
 scores(1045,728,LILAC);txt(1050,713,'sC(Q)',16)
else:
 txt(557,657,'C  Coverage Sugeno + historical review',18,anchor='start',bold=True)
 gaussian(545,737,61,31)
 txt(577,770,'Sugeno base',13,color=EDGE)
 arr(616,728,654,728)
 plane(673,710,4,5,7,'history',1)
 txt(686,770,'[Q, sbase]',13,color=EDGE)
 path('M714 728 H741',arrow=True)
 bar(747,704,23,53,'Review',TEAL)
 txt(775,784,'RBF · recurrence · full-prefix',12,color=EDGE)
 # Revealed historical records enter the review in addition to the current base score.
 plane(688,682,3,3,6,'history',2)
 txt(669,692,'Hₜ',13,anchor='end')
 path('M720 689 H760 V700',arrow=True)
 path('M785 728 H851 V741 H912',arrow=True)
 path('M851 728 V691 H893',arrow=True)
 tiny_vector(888,682,['#f4eef2',MAUVE,'#f4eef2',MAUVE],8)
 txt(949,670,'stable repair',13,color=EDGE)
 path('M925 696 V715',arrow=True)
 polygon([(915,711),(915,750),(946,730)],'#eee5ed',EDGE,.9)
 txt(970,770,'conditional selection',13,color=EDGE)
 arr(949,728,1025,728)
 scores(1045,701,LILAC);txt(1050,687,'sC(Q)',16)

# The output wiring is a three-to-one orthogonal collector, matching the reference.
path('M1067 451 H1111 V584' if refined else 'M1067 451 H1111 V584 H1140')
path('M1067 584 H1144',arrow=True)
if refined: circle(1111,584,2,INK,INK)
path('M1067 755 H1111 V584' if reviewed else 'M1067 728 H1111 V584')
bar(1147,548,26,73,'Mean',GREEN)
arr(1178,584,1209,584)
bar(1217,548,26,73,'Sign', '#c3b5d3')
path('M1252 584 H1271 V548 H1291',arrow=True)
path('M1271 584 V629 H1291',arrow=True)
txt(1280,535,'≥ 0',13)
txt(1280,655,'< 0',13)
circle(1335,548,18,'#f2e6ec',MAUVE);txt(1335,554,'M',18)
circle(1335,629,18,'#e9edf3',BLUE);txt(1335,635,'B',18)
txt(1335,581,'Malignant',14)
txt(1335,664,'Benign',14)
txt(1200,704,'s(Q) = (sA + sB + sC) / 3' if refined else '(sA + sB + sC) / 3',15)

# A compact protocol band uses the same matrix and tree language as the main panels.
if reviewed:
 if compact:
  rect(30,921,713,145,'white',EDGE,.8,'10 6')
  rect(777,921,633,145,'white',EDGE,.8,'10 6')
 if online_loop:
  path('M330 869 V917',INK,1.4,True)
  txt(342,886,'Online batch: predictions + base scores',16,anchor='start')
  path('M650 917 V869',INK,1.4,True,'4 3')
  txt(638,910,'Updated state for next batch',16,anchor='end')
 elif feedback:
  path('M330 869 V917',INK,1.4,True)
  txt(342,895,'Predictions + base scores',16,anchor='start')
  path('M542 917 V869',QUERY_INK,1.4,True,'4 3')
  txt(554,895,'Next-batch model + history',16,anchor='start',color=QUERY_INK)
 txt(35,906,'c  Predict before updating',20,anchor='start',bold=True)
 for x,label in [(92,'Predict Qₜ'),(246,'Reveal labels'),(403,'Insert + remine'),(548,'Refit model'),(694,'Predict Qₜ₊₁')]:
  txt(x,1020,label,17)
 plane(75,960,5,4,6,'history')
 plane(228,960,5,4,6,'history')
 for i in range(4):circle(267,962+i*6,2,MAUVE,'none')
 tree(402,947,.36,True)
 plane(530,960,4,4,6,'incidence',2)
 plane(676,960,5,4,6,'history')
 for x1,x2 in [(114,216),(281,362),(451,519),(568,664)]:arr(x1,973,x2,973)
 if history:
  txt(92,1043,'Save base score' if compact else 'Save coverage score',15,color=EDGE)
  path('M309 973 V1047 H333',EDGE,1.1,True)
  txt(341,1052,'Append labeled record and score to Hₜ₊₁',16,anchor='start')
 if not compact: path('M760 891 V1056' if history else 'M760 891 V1037',LIGHT,1)
 txt(790,906,'d  Fixed holdout within each split',20,anchor='start',bold=True)
 txt(790,941,'Training samples' if publication else 'Checkpoint models',18 if publication else 16,anchor='start')
 for x,n in zip([970,1080,1190,1300],[300,500,700,849]):
  txt(x,941,str(n),18 if publication else 16)
  plane(x-8,951,2,2,7,'incidence')
  path(f'M{x} 970 V983')
 path('M970 983 H1300')
 path('M1125 983 V1004',arrow=True)
 txt(1146,1001,'Score using panel b' if scoring_clarified else 'Evaluate on',15,anchor='start')
 plane(1109,1015,5,3,6,'history')
 txt(1149,1030,'Fixed query: 213 holdout records' if publication else 'Same 213 holdout records',16,anchor='start')
else:
 txt(35,829,'c  Predict before updating',18,anchor='start',bold=True)
 plane(78,870,5,4,6,'history',1)
 txt(98,927,'Save predictions',14)
 arr(131,884,196,884)
 plane(229,870,5,4,6,'history',1)
 for i in range(4):circle(268,872+i*6,1.8,MAUVE,'none')
 txt(249,927,'Reveal labels',14)
 arr(286,884,354,884)
 tree(414,855,.36,True)
 txt(415,927,'Insert + remine',14)
 arr(470,884,532,884)
 plane(568,870,4,4,6,'incidence',2)
 txt(589,927,'Refit next model',14)
 path('M632 884 H685 V845 H98 V854',EDGE,.9,True,'3 3')
 
 
 path('M760 817 V936',LIGHT,.8)
 txt(790,829,'d  Fixed holdout within each split',18,anchor='start',bold=True)
 if refined: txt(790,856,'Training records',13,anchor='start',color=EDGE)
 for x,n in zip([945,1060,1175,1290] if refined else [838,976,1114,1252],[300,500,700,849]):
  txt(x,856,str(n),13,color=EDGE)
  plane(x-9,870,2,2,6,'incidence',1)
  path(f'M{x} 887 V902')
 if refined:
  path('M945 902 H1290')
  path('M1117 902 V916',arrow=True)
 else:
  path('M838 902 H1252 M1045 902 V913',arrow=True)
 plane(1104 if refined else 1031,925,5,2,5,'history',1)
 txt(1196,937,'Same 213 records',14)

if publication: out('</g>')
out('</g></svg>')
f=OUT/('fig01_three_way_fusion_en_scoring_clarified.svg' if scoring_clarified else 'fig01_three_way_fusion_en_online_loop.svg' if online_loop else 'fig01_three_way_fusion_en_feedback.svg' if feedback else 'fig01_three_way_fusion_en_compact.svg' if compact else 'fig01_three_way_fusion_en_history.svg' if history else 'fig01_three_way_fusion_en_query_protocol.svg' if publication else 'fig01_three_way_fusion_en_dependencies.svg' if dependencies else 'fig01_three_way_fusion_en_reviewed.svg' if reviewed else 'fig01_three_way_fusion_en_layout_refined.svg' if refined else 'fig01_three_way_fusion_en_mfmr_architecture.svg')
f.write_text(''.join(S),encoding='utf-8')
print('Wrote vector figure')
