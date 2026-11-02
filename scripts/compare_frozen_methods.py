"""Supplementary replay of frozen morphology update, scored against saved v1.

No selection or detector editing is performed. Original test feedback informed
the update, so this is explicitly a post-development paired comparison.
"""
import argparse,csv,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from face_pixel.part_b import prepare_frozen_view,compare_validation
from face_pixel.batch import evaluate_predictions
from face_pixel.data_io import read_dataset_manifest,read_binary_mask,read_prediction_manifest
from face_pixel.evaluation import COUNT_NAMES,METRIC_NAMES,evaluate,summarize
from face_pixel.assessment import boundary_iou,paired_bootstrap_f1,paired_mean_interval
from face_pixel.visualization import _font_path

REVISION='40e7b48439e6eb95ba1825ea8c76ff391f2fc5a1'
CONFIG_HASHES={'v1':'c62adf481aa37212664ffaeed824d9f450e91d88baa7b8d6a7d4c5305cff34ba','v2':'239c9849d6779b65a7559c132534a33093a84882c246fe442d58eaf868fde994'}
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvsave(p,rows):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def predict(root,config,split,out):
    env=os.environ.copy();env.update(PYTHONPATH=str(root/'src'),PYTHONNOUSERSITE='1',PYTHONIOENCODING='utf-8')
    subprocess.run([sys.executable,'-m','face_pixel.validation','predict','--manifest',str(root/'data/processed/part_b/manifest.csv'),'--config',str(config),'--split',split,'--output',str(out)],env=env,cwd=root,check=True,capture_output=True)
    run=load(out/'run.json');assert run['ground_truth_read'] is False and run['metrics_computed'] is False
    assert run['runtime']==load(config)['runtime']
def board(e,path,font):
    normal=ImageFont.truetype(font,29);small=ImageFont.truetype(font,25)
    tile=420;gap=22;top=100;rowheight=472
    canvas=Image.new('RGB',(3*tile+4*gap,2*rowheight+top+65),'white');d=ImageDraw.Draw(canvas)
    d.text((gap,12),e['sample_id']+'  新旧方法对照',font=normal,fill='#18212D')
    d.text((gap,55),f"F1 {e['v1_metrics']['f1']:.3f} → {e['v2_metrics']['f1']:.3f}    Recall {e['v1_metrics']['recall']:.3f} → {e['v2_metrics']['recall']:.3f}",font=small,fill='#25344A')
    def error(pred):
        rgb=e['rgb'];gt=e['gt'].astype(bool);p=pred.astype(bool);v=e['valid'].astype(bool);arr=rgb.astype(float)*.35
        for cond,color in ((p & gt & v,(35,175,95)),(p & ~gt & v,(235,66,65)),(~p & gt & v,(45,105,240))):arr[cond]=rgb[cond]*.28+np.array(color)*.72
        return Image.fromarray(arr.astype(np.uint8))
    pics=[('原图',Image.fromarray(e['rgb'])),('原方法预测',Image.fromarray(e['v1']*255)),('新方法预测',Image.fromarray(e['v2']*255)),('人工真值',Image.fromarray(e['gt']*255)),('原方法误差',error(e['v1'])),('新方法误差',error(e['v2']))]
    for i,(label,pic) in enumerate(pics):
        r,c=divmod(i,3);x=gap+c*(tile+gap);y=top+r*rowheight
        d.text((x,y),label,font=normal,fill='#25344A');canvas.paste(pic.convert('RGB').resize((tile,tile),Image.Resampling.NEAREST),(x,y+43))
    d.text((gap,canvas.height-52),'绿色为正确检出，红色为误检，蓝色为漏检。',font=small,fill='#25344A');canvas.save(path)
def run(args):
    root,manifest,old,out=map(lambda p:Path(p).resolve(),(args.detector_root,args.manifest,args.original_run,args.output))
    assert not out.exists() or args.resume,'Use a new output directory or --resume to verify saved intermediate outputs'
    cfgs={'v1':root/'configs/morphology_reference.json','v2':root/'configs/final_config.json'}
    for name,path in cfgs.items():assert digest(path)==CONFIG_HASHES[name],f'Frozen configuration changed: {name}'
    font=_font_path(args.font)
    conf={k:load(p) for k,p in cfgs.items()};v2=conf['v2']
    assert v2['labels']==conf['v1']['labels']
    assert v2['selection']['prior_test_feedback'] is True and v2['selection']['independent_test_claim'] is False
    assert v2['threshold']['k']==conf['v1']['threshold']['k']==1.25
    for c in conf.values():
        assert c['selection']['fixed'] and c['selection']['split']=='validation' and not c['selection']['test_used']
        assert digest(root/'configs/morphology_plan.json')==c['selection']['plan_sha256']
        for name,wanted in c['source_sha256'].items():assert digest(root/'src/face_pixel'/name)==wanted,name
    if (root/'remote_snapshot.json').exists():
        snapshot=load(root/'remote_snapshot.json');assert snapshot['revision']==REVISION
        for f in snapshot['files']:
            data=(root/f['path']).read_bytes()
            assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==f['git_sha'],f['path']
    else:
        assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==REVISION
    oldreport=load(old/'independent_test.json');assert digest(manifest)==oldreport['c_manifest_sha256']
    data_check=prepare_frozen_view(root,manifest)
    out.mkdir(parents=True,exist_ok=args.resume);summaries={};equal={};entries=[]
    for split in ('validation','test'):
        for method,cp in cfgs.items():
            pred=out/'predictions'/f'{split}-{method}'
            if not pred.exists():predict(root,cp,split,pred)
            saved=load(pred/'run.json');assert saved['config_sha256']==digest(cp) and saved['runtime']==conf[method]['runtime']
            ev=out/'evaluation'/f'{split}-{method}'
            if ev.exists():
                assert args.resume
                summary=load(ev/'summary.json')
                paths=read_prediction_manifest(pred/'predictions.csv')
                dataset={r['sample_id']:r for r in read_dataset_manifest(manifest)}
                for inp in summary['provenance']['inputs']:
                    sid=inp['sample_id'];row=dataset[sid]
                    for path,key in ((paths[sid],'pred_sha256'),(row['image_path'],'image_sha256'),(row['mask_path'],'gt_sha256'),(row['valid_mask_path'],'valid_sha256')):assert digest(path)==inp[key]
            else:summary=evaluate_predictions(manifest,pred/'predictions.csv',ev,split=split,config_id=conf[method]['config_id'],run_config_path=cp,pred_encoding='binary255')
            if split=='validation':
                expected=root/('artifacts/part_b_morphology_v2/validation_candidates/candidate_01/summary.json' if method=='v1' else 'artifacts/part_b_morphology_v2/validation/summary.json')
                compare_validation(summary,load(expected))
            if method=='v1':
                current=read_prediction_manifest(pred/'predictions.csv');previous=read_prediction_manifest(old/f'predictions/{split}-final/predictions.csv')
                assert set(current)==set(previous)
                assert all(np.array_equal(read_binary_mask(current[s],'binary255'),read_binary_mask(previous[s],'binary255')) for s in current)
                equal[split]=len(current)
                assert summary['global']==oldreport['results'][f'{split}-final']['global']
            summaries[f'{split}-{method}']={k:summary[k] for k in ('n_images','global','image_mean')}
            print(split,method,summary['global'],flush=True)
        # Both validation comparisons have passed before test inference begins.
    predictions={k:read_prediction_manifest(out/f'predictions/test-{k}/predictions.csv') for k in cfgs}
    raw_by_id={r['sample_id']:r for r in csv.DictReader((root/'data/split_manifest.csv').open(encoding='utf-8-sig'))}
    semantic={i:{'pixels':0,'v1_predicted':0,'v2_predicted':0,'added':0,'removed':0} for i in range(19)}
    rows=[]
    for row in read_dataset_manifest(manifest):
        if row['split']!='test':continue
        sid=row['sample_id'];gt=read_binary_mask(row['mask_path']);valid=read_binary_mask(row['valid_mask_path']);rgb=np.array(Image.open(row['image_path']).convert('RGB'))
        e={'sample_id':sid,'gt':gt,'valid':valid,'rgb':rgb}
        r={'sample_id':sid}
        for k in cfgs:
            e[k]=read_binary_mask(predictions[k][sid],'binary255');e[k+'_metrics']=evaluate(e[k],gt,valid)
            r.update({f'{k}_{m}':val for m,val in e[k+'_metrics'].items()})
            r[f'{k}_boundary_iou']=boundary_iou(e[k],gt,valid)['boundary_iou']
        for metric in METRIC_NAMES:r['delta_'+metric]=r['v2_'+metric]-r['v1_'+metric]
        added=(e['v2']==1)&(e['v1']==0)&(valid==1);removed=(e['v2']==0)&(e['v1']==1)&(valid==1)
        r.update(added_tp=int(np.count_nonzero(added&(gt==1))),added_fp=int(np.count_nonzero(added&(gt==0))),removed_tp=int(np.count_nonzero(removed&(gt==1))),removed_fp=int(np.count_nonzero(removed&(gt==0))))
        raw=np.array(Image.open(root/raw_by_id[sid]['label_path']))
        for i in semantic:
            m=(raw==i)&(valid==1);s=semantic[i];s['pixels']+=int(m.sum());s['v1_predicted']+=int((m&(e['v1']==1)).sum());s['v2_predicted']+=int((m&(e['v2']==1)).sum());s['added']+=int((m&added).sum());s['removed']+=int((m&removed).sum())
        rows.append(r);entries.append(e)
    for k in cfgs:assert summarize(e[k+'_metrics'] for e in entries)['global']==summaries[f'test-{k}']['global']
    counts=lambda k:np.array([[r[f'{k}_{m}'] for m in ('tp','fp','tn','fn')] for r in rows],dtype=np.int64)
    report={'scope':'supplementary post-development comparison on original test split','independent_test_claim':False,'prior_test_feedback':True,'feedback_sample_id':'face_segment_test_g2_00570','detector_revision':REVISION,'original_revision':oldreport['b_revision'],'manifest_sha256':digest(manifest),'configs_sha256':{k:digest(p) for k,p in cfgs.items()},'source_sha256':v2['source_sha256'],'runtime':v2['runtime'],'view_check':data_check,'validation_replay':'exact counts and metrics match published reference and updated validation','v1_prediction_pixel_equality':equal,'configuration_changed_after_evaluation':False,'results':summaries,'test_delta':{m:summaries['test-v2']['global'][m]-summaries['test-v1']['global'][m] for m in (*COUNT_NAMES,*METRIC_NAMES)},'image_f1_changes':{'improved':sum(r['delta_f1']>1e-12 for r in rows),'declined':sum(r['delta_f1']< -1e-12 for r in rows),'unchanged':sum(abs(r['delta_f1'])<=1e-12 for r in rows)},'boundary_iou':{k:float(np.mean([r[f'{k}_boundary_iou'] for r in rows])) for k in cfgs},'pixel_changes':{m:sum(r[m] for r in rows) for m in ('added_tp','added_fp','removed_tp','removed_fp')},'bootstrap_global_f1':paired_bootstrap_f1(counts('v1'),counts('v2')),'bootstrap_image_mean_f1':paired_mean_interval([r['v1_f1'] for r in rows],[r['v2_f1'] for r in rows])}
    roles={'feedback':'face_segment_test_g2_00570','most_fn_recovered':max(rows,key=lambda r:r['added_tp']-r['removed_tp'])['sample_id'],'most_fp_added':max(rows,key=lambda r:r['added_fp']-r['removed_fp'])['sample_id'],'largest_f1_decline':min(rows,key=lambda r:r['delta_f1'])['sample_id']}
    report['case_selection']=roles
    (out/'figures').mkdir(exist_ok=args.resume);(out/'cases').mkdir(exist_ok=args.resume)
    for e in entries:board(e,out/'cases'/f"{e['sample_id'].rsplit('_',1)[-1]}.png",font)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(font);plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'svg.fonttype':'path'})
    fig,ax=plt.subplots(1,2,figsize=(11.4,4.3),layout='constrained')
    x=np.array([r['v1_f1'] for r in rows]);y=np.array([r['v2_f1'] for r in rows])
    for mask,color,label in ((y>x+1e-12,'#087E8B','F1 提高'),(y<x-1e-12,'#B87432','F1 下降'),(abs(y-x)<=1e-12,'#777777','F1 相同')):
        ax[0].scatter(x[mask],y[mask],color=color,s=32,label=f'{label} {mask.sum()} 张')
    ax[0].plot([0,1],[0,1],'--',color='#999999',lw=1);ax[0].set(xlabel='原方法逐图 F1',ylabel='新方法逐图 F1',xlim=(0,1),ylim=(0,1));ax[0].legend(fontsize=9)
    ax[1].scatter([100*r['delta_recall'] for r in rows],[100*r['delta_precision'] for r in rows],s=32,color='#087E8B');ax[1].axhline(0,color='#999999',ls='--',lw=1);ax[1].axvline(0,color='#999999',ls='--',lw=1);ax[1].set(xlabel='Recall 变化，百分点',ylabel='Precision 变化，百分点')
    for sid in set(roles.values()):
        i=next(i for i,r in enumerate(rows) if r['sample_id']==sid);ax[0].annotate(sid.rsplit('_',1)[-1],(x[i],y[i]),xytext=(5,4),textcoords='offset points',fontsize=8)
    for a in ax:a.grid(alpha=.2)
    for ext in ('png','svg'):fig.savefig(out/'figures'/f'paired_changes.{ext}',dpi=230)
    plt.close(fig)
    csvsave(out/'per_image_comparison.csv',rows);csvsave(out/'semantic_comparison.csv',[{'label_id':i,**v} for i,v in semantic.items()]);save(out/'comparison.json',report)
    save(out/'input_provenance.json',{'original_record_sha256':digest(old/'independent_test.json'),'canonical_audit_sha256':digest(manifest.parent/'audit.json'),'prediction_inputs':{name:load(out/'evaluation'/name/'summary.json')['provenance'] for name in summaries}})
    print(json.dumps({k:report[k] for k in ('test_delta','image_f1_changes','pixel_changes','boundary_iou','case_selection')},ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--detector-root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--original-run',required=True);p.add_argument('--output',required=True);p.add_argument('--font',help='Chinese font path, auto-detected on Windows and Linux');p.add_argument('--resume',action='store_true',help='Verify and reuse saved intermediate predictions after an interrupted analysis');run(p.parse_args())
