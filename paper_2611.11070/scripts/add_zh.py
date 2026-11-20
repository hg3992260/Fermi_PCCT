import os, re

base = r'I:\Fermi\paper_2611.11070'
path = os.path.join(base, 'paper.md')
doc = open(path, encoding='utf-8').read()

add = {
'S012': '**中文：**式 (3) 给出超软 d 通道的投影子强度与模亏损：D_ij 是赝轨道在 (ε_j − *T* − *V*_loc) 作用下的矩阵元再加上 ε_jQ_ij；Q_ij 是全电子轨道与赝轨道在芯区内模之差。i, j ∈ {1,2} 对应两个参考能量。',
'S029': '**中文：**式 (4) 是种类势的余弦基展开：n = 0…11 的主项覆盖到 8 Å⁻¹，n = 0…25 的高 q 尾部覆盖到 17 Å⁻¹；窗函数 ½(1+cos πx) 保证在 q = q_c 处平滑截断。',
'S032': '**中文：**式 (5) 是 s/p 的非局域项：对每个 s/p 原子、每个角动量通道 *l*（至 *l*_max(Z_a)）、每个径向函数 *r* = 1, 2 求和；强度是元素种类强度 D^Z_lr 加逐原子修正 ΔD_a,lr，角向部分由球谐 *m* 求和给出。',
'S037': '**中文：**式 (6) 即广义 Hellmann–Feynman 定理：能带 E_nk 对模型参数 θ 的导数，等于本征矢左乘 (∂*H*/∂θ − E_nk∂*S*/∂θ) 再右乘本征矢，归一化条件为 **c**†*S***c** = 1。这是"梯度穿过本征值求解器"的数学基础。',
'S102': '**中文：**式 (S1) 是有隙结构的损失：第一项是窗口 W_c 内能带误差的加权平均（带边两条能带权重 w_n = 5），第二项是带隙的绝对误差，第三项是全部占据带的弱正则项（λ_occ = 0.3）。',
'S103': '**中文：**式 (S2) 是金属的损失：先对窗口 M_c 内的预测取一个不参与求导的刚性平移 s_c（中位数），再计算平均绝对误差，且不含带隙项。',
'S113': '**中文：**式 (S6) 定义 d 元素的局域势：r_loc 以内是偶多项式 a′ + b′r² + c′r⁴，以外是中性原子的全电子自洽势 V_AE(r)；两者在 r_loc 处的数值、斜率与曲率连续。',
'S114': '**中文：**式 (S7) 用 ε_i 与 V_loc 构造 χ_i，再由矩阵 B_ij 求双正交化的投影子 β_i；式 (S8) 给出模亏损 Q_ij 与投影子强度 D_ij = B_ij + ε_jQ_ij。所有积分都在 r_cut = max(r^d_c, r_loc) + 0.3 Å 以内。',
'S115': '**中文：**式 (S9) 是 d 通道的最终形式：*V*_d = Σ\|β_i⟩D_ij⟨β_j\|，*S* = 1 + Σ\|β_i⟩Q_ij⟨β_j\|。其中 T_l = −(ħ²/2m)[d²/dr² − *l*(*l*+1)/r²] 为径向动能算符，Q₁₁ = q_d（模型中 Q 用其半正定部分替代，变化小于 0.003）。',
}

for aid, txt in add.items():
    a = f'<a id="{aid}"></a>'
    i = doc.index(a)
    nxt = doc.find('\n<a id=', i + len(a))
    assert nxt > 0, aid
    doc = doc[:nxt].rstrip('\n') + '\n\n' + txt + '\n' + doc[nxt:]

open(path, 'w', encoding='utf-8').write(doc)

n_orig = doc.count('**Original:**')
n_zh = doc.count('**中文：**')
n_cap = doc.count('**中文图注：**')
print('chars', len(doc), 'Original', n_orig, '中文', n_zh, '中文图注', n_cap)

for m in re.finditer(r'<a id="(S\d{3})"></a>', doc):
    aid = m.group(1)
    start = m.end()
    nxt = doc.find('<a id=', start)
    seg = doc[start:nxt if nxt > 0 else len(doc)]
    if '**中文' not in seg:
        print('NO-ZH', aid)
for m in re.finditer(r'<a id="([FT]\d{3})"></a>', doc):
    aid = m.group(1)
    start = m.end()
    nxt = doc.find('<a id=', start)
    seg = doc[start:nxt if nxt > 0 else len(doc)]
    if '**中文' not in seg:
        print('NO-ZH-CAP', aid)
