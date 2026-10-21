"""Plots of measured distributions and actual train30 images only."""
import json
import os
from pathlib import Path
import numpy as np
from geometry import resolve_profile, relation


def draw_distance_distributions(output, dist):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path = Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/msyh.ttc'
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    names = {'same_1_2':'同轨迹间隔1–2帧','same_3_8':'同轨迹间隔3–8帧',
             'same_9_plus':'同轨迹间隔>8帧','cross':'跨轨迹'}
    fig,axes = plt.subplots(2,2,figsize=(11,8))
    for ax,key,label in zip(axes.flat,['d_p','d_T','d_theta','D'],
            ['推杆距离（世界坐标）','T块位置距离（世界坐标）','周期角度差（度）','联合距离 D']):
        for kind,legend in names.items():
            values = dist[kind][key]
            edges = np.array(values['histogram_edges'][1:])
            if key == 'd_theta':
                edges = np.rad2deg(edges)
            ax.plot(edges,np.cumsum(values['histogram_counts'])/values['count'],label=legend)
        ax.set(xlabel=label,ylabel='累计样本对比例',ylim=(0,1))
        if key == 'D':
            ax.set_xscale('symlog',linthresh=.001)
        ax.set_xlim(left=0)
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('ALT-30 所有训练帧对：几何距离与时间间隔（无序帧对）')
    fig.tight_layout()
    destination = output/'figures/distance_distributions.png'
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=150)
    plt.close(fig)


def make_figures(output, config, d, pools, ep, local, source, root, dist, nn, cal, profiles, batches):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path = Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/msyh.ttc'
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    folder = output/'figures'
    folder.mkdir(parents=True,exist_ok=True)
    names = {'same_1_2':'同轨迹间隔1–2帧', 'same_3_8':'同轨迹间隔3–8帧',
             'same_9_plus':'同轨迹间隔>8帧','cross':'跨轨迹'}
    pool_names = {'all':'全部其他帧','exclude_local8':'排除同轨迹±8帧','cross':'仅跨轨迹'}
    draw_distance_distributions(output,dist)

    fig,axes = plt.subplots(2,2,figsize=(11,8))
    ks = config['neighbors']['top_k']
    bottom = np.zeros(len(ks))
    for kind,legend in names.items():
        values = [nn['all']['top_k'][str(k)]['temporal_relation_fractions'][kind]*100 for k in ks]
        axes[0,0].bar([str(k) for k in ks],values,bottom=bottom,label=legend)
        bottom += values
    axes[0,0].set(xlabel='Top-k',ylabel='近邻关系比例（%）',ylim=(0,100),title='允许全部近邻：来源构成')
    axes[0,0].legend(fontsize=8)
    for ax,key,label in zip([axes[0,1],axes[1,0],axes[1,1]],['d_p','d_T','d_theta'],
                           ['推杆距离均值','T块位置距离均值','角度差均值（度）']):
        for pool,legend in pool_names.items():
            values = [nn[pool]['top_k'][str(k)]['distance_components'][key]['mean'] for k in ks]
            if key == 'd_theta':
                values = np.rad2deg(values)
            ax.plot(ks,values,marker='o',label=legend)
        ax.set(xlabel='Top-k',ylabel=label,xticks=ks)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('几何近邻：来源与误差；近邻存在不等于满足几何阈值')
    fig.tight_layout()
    fig.savefig(folder/'nearest_neighbors.png',dpi=150)
    plt.close(fig)

    fig,axes = plt.subplots(3,3,figsize=(12,9),sharey=True)
    profile_names = list(profiles)
    for i,role in enumerate(['similar','hard_a','hard_b']):
        for j,pool in enumerate(pools):
            values = [profiles[p]['roles'][role][pool]['anchor_coverage_fraction']*100 for p in profile_names]
            axes[i,j].plot(profile_names,values,marker='o')
            for x,y in enumerate(values):
                axes[i,j].annotate(f'{y:.1f}%',(x,y),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
            axes[i,j].set(title=role+' / '+pool_names[pool],ylim=(0,110),ylabel='Anchor至少有1个候选（%）')
            axes[i,j].grid(alpha=.2)
    fig.suptitle('预先指定的三档阈值：候选覆盖率（完整57条件网格见JSON/CSV）')
    fig.tight_layout()
    fig.savefig(folder/'candidate_coverage.png',dpi=150)
    plt.close(fig)

    fig,axes = plt.subplots(2,3,figsize=(15,8))
    descriptions = [('completion_fraction','32帧完整率（随机组无角色配额）'),
        ('similar_eligible_unique_pairs','Similar有效不同帧对数'),
        ('hard_a_eligible_unique_pairs','Hard A有效不同帧对数'),
        ('hard_b_eligible_unique_pairs','Hard B有效不同帧对数'),
        ('episode_count','不同Episode数'),('duplicate_frame_fraction','重复帧比例')]
    for ax,(key,title) in zip(axes.flat,descriptions):
        for offset,(mode,label) in enumerate([('allow_repeated','允许重复'),('unique_frames','帧唯一'),('random_baseline','完全随机')]):
            group = [next(g for g in batches['groups'] if g['profile']=='primary' and g['pool']==pool and g['uniqueness']==mode) for pool in pools]
            values = [g[key] if key=='completion_fraction' else g['metrics_all_attempts'][key]['mean'] for g in group]
            ax.bar(np.arange(3)+(offset-1)*.24,values,width=.24,label=label)
        ax.set(xticks=range(3),xticklabels=list(pool_names.values()),title=title)
        ax.tick_params(axis='x',labelsize=8)
        ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('主阈值方案：100次模拟均值（含缺槽Batch，完成批次另列统计）')
    fig.tight_layout()
    fig.savefig(folder/'batch_comparison.png',dpi=150)
    plt.close(fig)

    example = config['figures']
    values = config['thresholds']['profiles'][example['profile']]
    threshold = resolve_profile(cal,values)
    eligible = {role:relation(d,role,threshold) & pools[example['pool']] for role in ['similar','hard_a','hard_b']}
    rng = np.random.default_rng(example['example_seed'])
    anchors = rng.choice(len(ep),example['example_anchors'],replace=False)
    fig,axes = plt.subplots(len(anchors),5,figsize=(15,3*len(anchors)),squeeze=False)
    examples = []
    for row,anchor in enumerate(anchors):
        selected = [int(anchor)]
        for role in ['similar','hard_a','hard_b']:
            candidates = np.flatnonzero(eligible[role][anchor])
            selected.append(int(rng.choice(candidates)) if len(candidates) else None)
        selected.append(int(rng.choice(np.delete(np.arange(len(ep)),anchor))))
        records = []
        for col,(title,index) in enumerate(zip(['Anchor','Similar','Hard A','Hard B','Random'],selected)):
            ax = axes[row,col]
            if index is None:
                ax.text(.5,.5,'无符合条件候选',ha='center',va='center',transform=ax.transAxes,color='darkred')
                ax.set_title(title)
                records.append({'role':title,'row_index':None})
            else:
                ax.imshow(root['data/img'][int(source[index])].astype(np.uint8),interpolation='nearest')
                ax.set_title(f'{title} | ep{ep[index]}:{local[index]}\np={d["d_p"][anchor,index]:.1f}, T={d["d_T"][anchor,index]:.1f}, θ={np.rad2deg(d["d_theta"][anchor,index]):.1f}°',fontsize=9)
                records.append({'role':title,'row_index':index,'episode_id':int(ep[index]),
                    'local_frame_index':int(local[index]),'source_index':int(source[index])})
            ax.axis('off')
        examples.append({'anchor':int(anchor),'selections':records})
    fig.suptitle('训练集固定随机Anchor示例；Similar/Hard仅跨轨迹；几何规则不代表动作相容')
    fig.tight_layout()
    fig.savefig(folder/'sampling_examples.png',dpi=150)
    plt.close(fig)
    (output/'results/figure_examples.json').write_text(json.dumps({'profile':example['profile'],
        'pool':example['pool'],'thresholds':threshold,'examples':examples},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
