"""Original ICDE figures. Quantitative marks use all recorded participant units.

Only abstract waveform/array glyphs in Figures 1 and 2 are schematic.
THINGSplus-CC0 photographs illustrate concepts in Figures 1 and 2.
Figure 6 pairs CC0 concept illustrations with verified candidate-record identities.
"""
from pathlib import Path
import os,sys,json,argparse,hashlib,csv
parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path)
parser.add_argument('--inputs',type=Path);parser.add_argument('--output',type=Path)
parser.add_argument('--paper-dir',type=Path)
parser.add_argument('--figures',type=int,nargs='+',default=[1,2,3,4,5,6])
parser.add_argument('--languages',nargs='+',choices=['en','zh'],default=['en','zh'])
args=parser.parse_args()
R=args.workspace or Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(R/'.cache/matplotlib')
if os.environ.get('FIGURE_QA_HELPERS'): sys.path.insert(0,os.environ['FIGURE_QA_HELPERS'])
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch,Ellipse,Circle
from matplotlib.font_manager import FontProperties,fontManager
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from PIL import Image,ImageOps
try:
 from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
 require_matplotlib_panel_alignment=None

OUT=args.output or R/'materials/final_figures';OUT.mkdir(exist_ok=True,parents=True)
summary_path=args.inputs/'summary.json' if args.inputs else R.parent/'data/results/compact_final_v87/summary.json'
case_path=args.inputs/'qualitative_cases_v151.json' if args.inputs else R/'materials/experiment_archive/results/qualitative_cases_v151.json'
PHOTO=args.inputs/'cc0' if args.inputs else R/'materials/cc0_illustrations'
S=json.loads(summary_path.read_text(encoding='utf-8'))
CASES=json.loads(case_path.read_text(encoding='utf-8'))['cases']
ME='hybrid_block_mlp';REF='anchor_nll'
BLUE='#35638B';PALE='#EAF1F7';PURPLE='#8A78A2';LAV='#F0ECF5';AMBER='#D5A458';CREAM='#FBF4E6';GRAY='#7C8893';LINE='#BCC7D1';INK='#263746'
LANG='en'
ZH={
 'Catalog and filter':'目录与筛选','Repeated query':'重复查询','Compare and weigh':'比较与加权','Calibrated result':'校准结果',
 'Concept: screw':'概念：螺丝','Ten record IDs':'十个记录编号','Four EEG recordings':'四次脑电记录','Repeat geometry':'重复记录几何','Two gallery anchors':'双图库锚点','Ranks + probabilities':'排序与概率',
 'Concept illustrations: THINGSplus-CC0; diagram symbols are schematic.':'概念配图来自 THINGSplus-CC0；图形符号为流程示意。',
 'Offline fitting and record cache':'离线拟合与记录缓存','Online query execution':'在线查询执行',
 'Training concepts':'训练概念','1,254 concepts':'1,254 个概念','Calibration concepts':'校准概念','100 concepts':'100 个概念',
 'EEG repeats':'脑电重复记录','Visual records':'图像记录','NICE / ATM-S + PCA':'NICE / ATM-S + PCA',
 'A/B contrasts':'A/B 差分','Repeat covariance':'重复协方差','Ridge + two MLPs':'岭回归 + 双 MLP',
 'Three score weights':'三个分数权重','Held-out NLL fit':'留出集 NLL 拟合',
 'Cached candidate state':'缓存候选状态','Frozen EEG encoders':'冻结脑电编码器','Query recordings':'查询记录',
 'Two encoders, six passes':'双编码器，六次编码','Repeat score':'重复度量分数','NICE anchor':'NICE 锚点','ATM-S anchor':'ATM-S 锚点',
 'Weighted sum + softmax':'加权求和 + softmax','Exact image IDs':'准确图像编号','Conditional probabilities':'条件概率',
 'Participant Top-1 gains':'参与者 Top-1 增益','Probability and selective risk':'概率质量与选择性风险',
 'Top-1 gain over NLL anchors (pp)':'相对 NLL 锚点的 Top-1 增益（百分点）',
 'Brier reduction':'Brier 降低量','AURC reduction':'AURC 降低量','All ten improve':'十人均改善','Both improve':'两项均改善',
 'NLL anchors':'NLL 锚点','Ours':'本文方法','Retained queries':'保留查询比例','Error among retained queries (%)':'保留查询的错误率（%）',
 'Covariance and ranking':'协方差与排序','Covariance and probability':'协方差与概率','Total':'总协方差','Residual':'预测残差','Permuted':'置乱重复','Diagonal':'重复对角','Repeat block':'重复分块',
 'Top-1 change vs. isotropic (pp)':'相对各向同性的 Top-1 变化（百分点）','Brier reduction vs. isotropic':'相对各向同性的 Brier 降低量',
 'Concept icon (CC0)':'概念配图（CC0）','Query record':'查询记录','Target rank among ten candidates':'目标在十个候选中的排名',
 'barrette':'发夹','screw':'螺丝','velcro':'魔术贴','nail':'钉子','combination lock':'密码锁','zipper':'拉链',
 'Rank 1: exact record recovered':'排名 1：准确找回目标记录',
 'Metric + templates + anchors':'度量 + 模板 + 锚点','4 EEG':'四次 EEG',
 'Paired retrieval accuracy':'配对检索准确率','Top-1 accuracy (%)':'Top-1 准确率（%）',
 'Ten participant pairs':'十位参与者配对','Participant mean':'参与者均值',
 '10/10 improve; mean +0.90 pp':'十人均改善；平均 +0.90 个百分点',
 'Both reductions are positive':'两项降低量均为正',
 'Query coverage (%)':'查询覆盖率（%）','Retained-query error (%)':'保留查询的错误率（%）',
 'Covariance-source effects by participant':'各参与者的协方差来源效应',
 'Top-1 change (pp)':'Top-1 变化（百分点）',
 '1000 x Brier reduction':'Brier 降低量乘以 1000','Mean':'均值',
}
def tr(s):return ZH.get(s,s) if LANG=='zh' else s
def config(lang):
 global LANG;LANG=lang
 family='Arial'
 if lang=='zh':
  fontManager.addfont('C:/Windows/Fonts/simhei.ttf');family=FontProperties(fname='C:/Windows/Fonts/simhei.ttf').get_name()
 mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':[family,'DejaVu Sans'],'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.4,'ytick.labelsize':7.4,'legend.fontsize':7.4,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'axes.unicode_minus':False,'legend.frameon':False,'figure.facecolor':'white','savefig.facecolor':'white'})
def text(a,x,y,s,size=8,**kw):return a.text(x,y,tr(s),ha=kw.pop('ha','center'),va=kw.pop('va','center'),fontsize=size,color=kw.pop('color',INK),**kw)
def panel(a,k,title):
 a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
 a.text(0,1.035,k,weight='bold',fontsize=9,va='bottom',color=INK)
 a.text(.085,1.035,tr(title),weight='bold',fontsize=8.2,va='bottom',color=INK)
def title(a,k,s):
 a.text(0,1.055,k,transform=a.transAxes,weight='bold',fontsize=9,va='bottom',color=INK)
 a.text(.07,1.055,tr(s),transform=a.transAxes,weight='bold',fontsize=8.2,va='bottom',color=INK)
def rect(a,x,y,w,h,fc=PALE,ec=LINE,dashed=False):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=0.02',fc=fc,ec=ec,lw=.7,linestyle='--' if dashed else '-'))
def arrow(a,p,q,style='-',color=GRAY):a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=8,lw=.8,color=color,linestyle=style,connectionstyle='arc3'))
def wave(a,x,y,w,h,n=4):
 # Abstract waveform glyphs; this is not an EEG result or a measured trace.
 t=np.linspace(0,1,100)
 for i in range(n):
  v=np.sin(2*np.pi*(2.1*t+.08*i))*.24+np.sin(2*np.pi*(7*t+.13*i))*.11
  a.plot(x+t*w,y+(i+.5)*h/n+v*h/n,lw=.55,color=[BLUE,PURPLE,GRAY,AMBER][i%4])
def grid(a,x,y,w,h,nr=4,nc=5):
 colors=[PALE,LAV,CREAM,'#D4E3F0','#DDD4E8']
 for i in range(nr):
  for j in range(nc):a.add_patch(Rectangle((x+j*w/nc,y+i*h/nr),w/nc*.91,h/nr*.86,fc=colors[(i*3+j)%len(colors)],ec=LINE,lw=.3))
def net(a,x,y,w,h):
 for ncol,nn in enumerate([3,4,3]):
  ys=np.linspace(y+.12*h,y+.88*h,nn);xx=x+ncol*w/2
  if ncol:
   prev=np.linspace(y+.12*h,y+.88*h,[3,4,3][ncol-1]);px=x+(ncol-1)*w/2
   for py in prev:
    for yy in ys:a.plot([px,xx],[py,yy],c=LINE,lw=.3,zorder=2)
  for yy in ys:a.add_patch(Circle((xx,yy),.014,ec=PURPLE,fc='white',lw=.55,zorder=3))
def photo(a,word,x,y,w,h):
 p=PHOTO/f'{word}.jpg'
 im=Image.open(p).convert('RGB')
 bounds=a.get_position();aspect=(bounds.width*a.figure.get_figwidth()*w)/(bounds.height*a.figure.get_figheight()*h)
 canvas=Image.new('RGB',(max(1,int(300*aspect)),300),'white');im=ImageOps.contain(im,canvas.size);canvas.paste(im,((canvas.width-im.width)//2,(canvas.height-im.height)//2))
 a.imshow(canvas,extent=(x,x+w,y,y+h),aspect='auto',zorder=3)
 a.add_patch(Rectangle((x,y),w,h,fc='none',ec=LINE,lw=.45,zorder=4))
def export(f,name,axes=None):
 folder=OUT/LANG;folder.mkdir(exist_ok=True);f.canvas.draw()
 if require_matplotlib_panel_alignment:
  require_matplotlib_panel_alignment(f,axes=axes,json_out=folder/(name+'.alignment.json'),overlay_svg=folder/(name+'.alignment.svg'),strict=True)
 else:
  (folder/(name+'.alignment.json')).write_text(json.dumps({'status':'not audited in this environment','reason':'optional geometry-audit helper is not installed'}),encoding='utf-8')
 f.savefig(folder/(name+'.pdf'));f.savefig(folder/(name+'.svg'));f.savefig(folder/(name+'.png'),dpi=300)
 paper=args.paper_dir or (R/'paper' if not args.inputs else None)
 if paper:
  dest=paper/('fig' if LANG=='en' else 'zh/fig');dest.mkdir(exist_ok=True,parents=True)
  (dest/(name+'.pdf')).write_bytes((folder/(name+'.pdf')).read_bytes())
 plt.close(f);print('EXPORTED',LANG,name,flush=True)
def part(method,metric):return np.asarray(S['participant']['within10'][method][metric],float)

def intro():
 f,aa=plt.subplots(1,4,figsize=(7.16,1.75));f.subplots_adjust(left=.018,right=.984,top=.79,bottom=.15,wspace=.18)
 for a,k,t in zip(aa,'abcd',['Catalog and filter','Repeated query','Compare and weigh','Calibrated result']):panel(a,k,t)
 a=aa[0]
 for j,w in enumerate(['barrette','screw','nail']):photo(a,w,.015+j*.335,.56,.30,.39)
 text(a,.5,.43,'Concept: screw',7.6,weight='bold');arrow(a,(.5,.35),(.5,.28))
 for j in range(10):
  x=.03+(j%5)*.194;y=.04+(j//5)*.095
  rect(a,x,y,.17,.08,CREAM);text(a,x+.085,y+.04,f'I{j+1}',6.5)
 text(a,.5,-.035,'Ten record IDs',7.4)
 a=aa[1];rect(a,.02,.13,.96,.8,fc='white');wave(a,.07,.19,.86,.68)
 text(a,.5,-.035,'Four EEG recordings',7.4)
 a=aa[2];rect(a,.02,.60,.96,.31,PALE);text(a,.5,.755,'Repeat geometry',7.5,weight='bold')
 rect(a,.02,.14,.96,.31,LAV);text(a,.5,.295,'Two gallery anchors',7.5);arrow(a,(.5,.57),(.5,.48))
 a=aa[3]
 for y,label,w in [( .79,'I3',.55),(.51,'I7',.32),(.23,'I2',.15)]:
  text(a,.08,y,label,7.5);a.add_patch(Rectangle((.23,y-.075),w,.15,fc=BLUE if label=='I3' else '#CEDCE8',ec='none'));text(a,.89,y,'p',7.5)
 text(a,.5,-.035,'Ranks + probabilities',7.4)
 f.text(.5,.025,tr('Concept illustrations: THINGSplus-CC0; diagram symbols are schematic.'),ha='center',fontsize=7,color=GRAY)
 export(f,'fig01_catalog_story')

def model():
 f,aa=plt.subplots(1,2,figsize=(7.16,3.65));f.subplots_adjust(left=.025,right=.98,top=.90,bottom=.045,wspace=.11)
 for a,k,t in zip(aa,'ab',['Offline fitting and record cache','Online query execution']):panel(a,k,t)
 a=aa[0];rect(a,.005,.005,.99,.99,fc='white',dashed=True)
 rect(a,.03,.80,.44,.16,PALE);text(a,.25,.91,'Training concepts',7.7,weight='bold');text(a,.25,.845,'1,254 concepts',7.1)
 rect(a,.53,.80,.44,.16,LAV);text(a,.75,.91,'Calibration concepts',7.4,weight='bold');text(a,.75,.845,'100 concepts',7.1)
 wave(a,.05,.60,.32,.14);text(a,.21,.77,'EEG repeats',7.3)
 photo(a,'screw',.43,.59,.18,.15);grid(a,.64,.60,.28,.13,3,5);text(a,.69,.77,'Visual records',7.3)
 arrow(a,(.21,.59),(.21,.54));arrow(a,(.70,.59),(.70,.54))
 rect(a,.035,.43,.36,.11,PALE);text(a,.215,.485,'NICE / ATM-S + PCA',7)
 rect(a,.48,.43,.47,.11,LAV);text(a,.715,.485,'Ridge + two MLPs',7.2)
 arrow(a,(.22,.42),(.22,.36));arrow(a,(.72,.42),(.72,.36));arrow(a,(.40,.485),(.47,.485))
 rect(a,.035,.19,.36,.17,CREAM);grid(a,.07,.22,.12,.10,4,4);text(a,.29,.30,'A/B contrasts',6.5);text(a,.215,.16,'Repeat covariance',7.2)
 rect(a,.48,.19,.47,.17,PALE);net(a,.57,.23,.29,.09);text(a,.715,.16,'Cached candidate state',7.1)
 arrow(a,(.21,.12),(.21,.075));arrow(a,(.72,.12),(.72,.075))
 text(a,.5,.052,'Metric + templates + anchors',8.2,weight='bold')
 a=aa[1];rect(a,.005,.005,.99,.99,fc='white',dashed=True)
 text(a,.23,.935,'Query recordings',7.8,weight='bold');wave(a,.05,.77,.35,.13)
 for j,label in enumerate(['A2','B2','R4']):
  rect(a,.49,.88-j*.068,.20,.055,PALE);text(a,.59,.907-j*.068,label,7)
 text(a,.83,.855,'4 EEG',7.3);arrow(a,(.59,.72),(.59,.68))
 rect(a,.05,.56,.90,.12,PALE);text(a,.5,.64,'Frozen EEG encoders',7.6,weight='bold');text(a,.5,.588,'Two encoders, six passes',7.1)
 arrow(a,(.5,.55),(.5,.50))
 for j,(label,fc) in enumerate([('Repeat score',CREAM),('NICE anchor',LAV),('ATM-S anchor',LAV)]):
  x=.035+j*.325;rect(a,x,.34,.285,.16,fc);text(a,x+.1425,.455,label,6.9)
  for k,h in enumerate([.035,.067,.05,.023]):a.add_patch(Rectangle((x+.04+k*.056,.36),.035,h,fc=BLUE if j==0 else PURPLE,ec='none'))
  arrow(a,(x+.1425,.33),(.5,.27))
 rect(a,.20,.16,.62,.11,PALE);text(a,.51,.215,'Weighted sum + softmax',7.6,weight='bold')
 arrow(a,(.51,.15),(.51,.105));text(a,.29,.055,'Exact image IDs',7);text(a,.75,.055,'Conditional probabilities',6.8)
 # Calibration is control flow, drawn through the shared gutter.
 p0=aa[0].transData.transform((.98,.87));p1=aa[1].transData.transform((.19,.215))
 inv=f.transFigure.inverted();p0=inv.transform(p0);p1=inv.transform(p1)
 f.lines.append(plt.Line2D([p0[0],.503,.503,p1[0]],[p0[1],p0[1],p1[1],p1[1]],transform=f.transFigure,color=PURPLE,lw=.8,linestyle='--'))
 f.add_artist(FancyArrowPatch((p1[0]-.015,p1[1]),(p1[0],p1[1]),transform=f.transFigure,arrowstyle='-|>',mutation_scale=8,color=PURPLE,lw=.8))
 f.text(.51,.50,tr('Three score weights'),rotation=90,rotation_mode='anchor',va='center',ha='left',fontsize=6.8,color=PURPLE)
 export(f,'fig02_retrieval_pipeline')

def performance():
 f,aa=plt.subplots(1,2,figsize=(7.16,2.75));f.subplots_adjust(left=.095,right=.98,top=.79,bottom=.23,wspace=.45)
 a=aa[0];title(a,'a','Paired retrieval accuracy')
 x=part(REF,'top1')*100;y=part(ME,'top1')*100
 assert len(x)==len(y)==10 and np.all(y>x)
 for u,v in zip(x,y):
  a.plot([0,1],[u,v],color='#B8C6D1',lw=.9,zorder=1)
 a.scatter(np.zeros(10),x,s=19,facecolors='white',edgecolors=GRAY,lw=.7,zorder=3)
 a.scatter(np.ones(10),y,s=20,c=BLUE,edgecolors='white',lw=.35,zorder=3)
 a.plot([0,1],[x.mean(),y.mean()],color=BLUE,lw=2.0,zorder=4)
 a.scatter([0,1],[x.mean(),y.mean()],s=40,marker='D',c=BLUE,edgecolors='white',lw=.6,zorder=5)
 a.set(xlim=(-.18,1.18),ylim=(32,46),xticks=[0,1],xticklabels=[tr('NLL anchors'),tr('Ours')],yticks=[32,36,40,44])
 a.set_ylabel(tr('Top-1 accuracy (%)'));a.grid(axis='y',lw=.4,alpha=.2)
 a.text(.5,1.19,tr('10/10 improve; mean +0.90 pp'),transform=a.transAxes,ha='center',fontsize=7.4,color=BLUE)
 a=aa[1];title(a,'b','Probability and selective risk');x=part(REF,'brier')-part(ME,'brier');y=part(REF,'aurc')-part(ME,'aurc')
 assert len(x)==len(y)==10 and np.all(x>0) and np.all(y>0)
 a.axvline(0,c=LINE,lw=.8);a.axhline(0,c=LINE,lw=.8);a.scatter(x,y,s=30,facecolors=BLUE,edgecolors='white',linewidths=.4,zorder=3)
 a.set(xlim=(-.001,max(.03,x.max()*1.16)),ylim=(-.002,max(.042,y.max()*1.18)));a.set_xlabel(tr('Brier reduction'));a.set_ylabel(tr('AURC reduction'))
 a.grid(lw=.4,alpha=.15);export(f,'fig03_retrieval_benefits')

def retention():
 f,a=plt.subplots(figsize=(3.48,2.55));f.subplots_adjust(left=.17,right=.98,top=.82,bottom=.22)
 levels=[25,50,75];positions=np.arange(3);width=.32
 # Zero-based bars show means. All ten units are preserved as observed points;
 # fixed horizontal offsets separate markers without changing any measurement.
 for offset,method,color,label in [(-.18,REF,'#D8E0E7','NLL anchors'),(.18,ME,BLUE,'Ours')]:
  values=[part(method,f'risk_at_{c}')*100 for c in levels]
  means=np.array([v.mean() for v in values]);centers=positions+offset
  a.bar(centers,means,width=width,color=color,edgecolor=GRAY if method==REF else BLUE,lw=.5,zorder=2,label=tr(label))
  for xx,v,mean in zip(centers,values,means):
   a.scatter(xx+np.linspace(-.10,.10,10),v,s=9,facecolors='white' if method==ME else GRAY,edgecolors=BLUE if method==ME else 'white',lw=.4,zorder=4)
   a.text(xx,7.0,f'{mean:.1f}',ha='center',va='center',fontsize=7.0,color=INK if method==REF else 'white',weight='bold',zorder=5)
 a.set(xticks=positions,xticklabels=[str(c)for c in levels],ylim=(0,70),yticks=[0,20,40,60])
 a.set_xlabel(tr('Query coverage (%)'));a.set_ylabel(tr('Retained-query error (%)'));a.grid(axis='y',lw=.4,alpha=.2,zorder=0)
 h,l=a.get_legend_handles_labels();f.legend(h,l,loc='upper center',bbox_to_anchor=(.57,1),ncol=2,fontsize=7.2)
 export(f,'fig04_selective_querying')

def covariance():
 f,aa=plt.subplots(2,1,figsize=(7.16,3.35));f.subplots_adjust(left=.145,right=.89,bottom=.12,top=.87,hspace=.76)
 methods=['compact_total','compact_residual','compact_permuted','compact_diagonal',ME];names=['Total','Residual','Permuted','Diagonal','Repeat block']
 cmap=LinearSegmentedColormap.from_list('SignedEffect',['#D8B67C','#FAFAF8','#5485A9'])
 for a,k,t,mult,metric in zip(aa,'ab',['Top-1 change (pp)','1000 x Brier reduction'],[100,-1000],['top1','brier']):
  title(a,k,t)
  raw=np.array([(part(m,metric)-part('compact_isotropic',metric))*mult for m in methods])
  assert raw.shape==(5,10) and np.isfinite(raw).all()
  values=np.column_stack([raw,raw.mean(axis=1)])
  limit=float(np.max(np.abs(values)))
  im=a.imshow(values,aspect='auto',cmap=cmap,norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),interpolation='nearest')
  for row in range(5):
   for col in range(11):
    a.text(col,row,f'{values[row,col]:.2f}',ha='center',va='center',fontsize=6.4,weight='bold' if col==10 else 'normal',color=INK)
  a.set(xticks=range(11),xticklabels=[f'S{i:02}'for i in range(1,11)]+[tr('Mean')],yticks=range(5),yticklabels=[tr(n)for n in names])
  a.tick_params(axis='both',length=0,pad=3,labelsize=7)
  for spine in a.spines.values():spine.set_visible(False)
  a.axvline(9.5,c=GRAY,lw=.7)
  a.add_patch(Rectangle((-.5,3.5),11,1,fc='none',ec=BLUE,lw=.9,clip_on=False))
  a.get_yticklabels()[-1].set_color(BLUE);a.get_yticklabels()[-1].set_weight('bold')
  bounds=a.get_position();cax=f.add_axes([.92,bounds.y0,.013,bounds.height]);bar=f.colorbar(im,cax=cax);bar.ax.tick_params(labelsize=6.5,length=2,pad=2);bar.outline.set_visible(False)
 export(f,'fig05_matched_geometry',axes=aa)

def cases():
 record=json.loads((args.inputs/'figure6_gallery_cases.json').read_text(encoding='utf-8'))
 f,aa=plt.subplots(6,1,figsize=(7.16,3.80));f.subplots_adjust(left=.018,right=.985,top=.955,bottom=.075,hspace=.32)
 short={'NICE_global':'N','ATM_global':'A',REF:'L',ME:'O'}
 for a,c in zip(aa,record['cases']):
  a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
  query=Path(c['query_id']).stem.rsplit('_',1)[-1]
  label=tr(c['concept'].replace('_',' '))if LANG=='zh'else c['concept'].replace('_',' ').capitalize()
  heading=f"{c['panel']}  {label}  |  "+('查询编号：'if LANG=='zh'else'Query ID: ')+query
  a.text(0,1.04,heading,ha='left',va='center',fontsize=7.1,color=INK,weight='bold')
  ranks=' / '.join(short[m]+str(v['target_rank'])for m,v in c['models'].items())
  a.text(1,1.04,('目标排名：'if LANG=='zh'else'Target ranks: ')+ranks,ha='right',va='center',fontsize=6.6,color=INK)
  photo(a,c['concept'],.002,.16,.095,.67)
  a.text(.0495,-.075,'CC0',ha='center',va='center',fontsize=6.0,color=GRAY)
  for k,item in enumerate(c['candidates']):
   x=.122+k*.088;width=.079
   a.add_patch(Rectangle((x,.16),width,.67,fc='#E9F4F1'if item['target']else'#F3F5F7',ec='#16877B'if item['target']else LINE,lw=1.1 if item['target']else .5))
   # Each card represents a record ID, not an image-content substitute.
   ident=Path(item['id']).stem.rsplit('_',1)[-1]
   a.text(x+width/2,.495,ident,ha='center',va='center',fontsize=7.3,color=INK,weight='bold'if item['target']else'normal')
   tag=' '.join(short[m]for m in item['selected_by'])
   if tag:a.text(x+width/2,-.075,tag,ha='center',va='center',fontsize=6.0,color=BLUE if 'O'in tag else INK,weight='bold'if 'O'in tag else'normal')
 footer=('照片：CC0 概念示意；卡片：真实候选编号。青框：目标。N：NICE；A：ATM-S；L：NLL 锚点；O：本文。'if LANG=='zh'else'Photos: CC0 concept illustrations; cards: candidate IDs. Teal: target. N: NICE; A: ATM-S; L: NLL anchors; O: ours.')
 f.text(.5,.015,footer,ha='center',va='center',fontsize=6.1,color=INK)
 export(f,'fig06_catalog_cases',axes=aa)

for language in args.languages:
 config(language)
 for index,fn in enumerate([intro,model,performance,retention,covariance,cases],start=1):
  if index in args.figures:fn()
rows=[]
for i in range(10):
 row={'participant':f'S{i+1:02}','top1_gain_pp':float((part(ME,'top1')[i]-part(REF,'top1')[i])*100),'brier_reduction':float(part(REF,'brier')[i]-part(ME,'brier')[i]),'aurc_reduction':float(part(REF,'aurc')[i]-part(ME,'aurc')[i])}
 rows.append(row)
with (OUT/'plotted_participant_differences.csv').open('w',encoding='utf-8',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'data_provenance.json').write_text(json.dumps({'summary_sha256':hashlib.sha256(summary_path.read_bytes()).hexdigest(),'participants':10,'seeds':[2028,2029,2030],'averaging':'Seeds within participant; participant is statistical unit','photographs':'Only THINGSplus-CC0 concept photographs; Figure 6 candidate contents are represented by verified record IDs, not substitute stimulus photos','schematic_symbols':'Figures 1 and 2 waveform and array glyphs; not measured EEG or model outputs','case_count':6,'ranks_unchanged':True,'performance_selection':'All ten participants; all five covariance alternatives; all fixed six cases'},ensure_ascii=False,indent=2),encoding='utf-8')
data=OUT/'plotted_data';data.mkdir(exist_ok=True)
rows=[]
for subject in range(10):
 for method in [REF,ME]:
  row={'participant':f'S{subject+1:02}','method':method,'top1_percent':float(part(method,'top1')[subject]*100),'brier':float(part(method,'brier')[subject]),'aurc':float(part(method,'aurc')[subject])}
  row.update({f'risk_at_{c}_percent':float(part(method,f'risk_at_{c}')[subject]*100)for c in [25,50,75]});rows.append(row)
with (data/'fig03_fig04_participant_values.csv').open('w',encoding='utf-8',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=[]
for method in ['compact_total','compact_residual','compact_permuted','compact_diagonal',ME]:
 for subject in range(10):rows.append({'participant':f'S{subject+1:02}','method':method,'top1_change_pp':float((part(method,'top1')[subject]-part('compact_isotropic','top1')[subject])*100),'brier_reduction_native':float(part('compact_isotropic','brier')[subject]-part(method,'brier')[subject])})
with (data/'fig05_covariance_participant_values.csv').open('w',encoding='utf-8',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
