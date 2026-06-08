# 需求基于QwenImage改写实现下述功能，功能描述如下：

按照论文的需求文档/home/sxm/data02Space/GraduationProject/textToLayoutBaseRADM/README.md改写
RADM仓库/home/sxm/data02Space/GraduationProject/textToLayoutBaseRADM/RADM的代码，添加相关代码实现

# 论文需求文档

\xsubsection{约束图表示与空间语义编码}{Constraint Graph Representation and Spatial-Semantic Encoding}

在前文语义解析阶段，本文已从用户输入中提取结构化设计要素及其空间关系约束。然而，这些信息仍以符号或离散关系形式存在，难以直接作为生成模型的条件输入。为在生成阶段有效利用这些约束，需将其转化为连续且可学习的特征表示。

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{基于异构图神经网络的语义约束编码框架图v2.png}
  \caption{基于异构图神经网络的语义约束编码框架图}
  \label{fig:semantic_constraint_encoding}
\end{figure}

为此，本文引入基于异构图神经网络的空间语义约束编码模块。如图~\ref{fig:semantic_constraint_encoding}所示，该模块通过构建多模态约束图，并利用关系感知图神经网络RGCN\cite{yuanRGNNRecurrentGraph2024}进行拓扑推理，将离散语义与空间关系统一映射为结构化节点表征。经该编码过程，模型可显式建模元素间的拓扑结构与空间语义关系，从而学习同时包含语义上下文与拓扑约束的中间表示。最终得到的节点表征作为结构化条件输入后续布局生成模型，在生成过程中显式引入语义与空间约束。



为将用户输入的非结构化设计意图转化为可计算的结构化表示，构建模态约束图$\mathcal{G} = (\mathcal{V}, \mathcal{E})$，其中$\mathcal{V}$与$\mathcal{E}$分别表示节点与边集合。图中节点对应视觉元素实例，边表示元素间空间约束关系。具体地，$\mathcal{V} = \mathcal{V}_{bg} \cup \mathcal{V}_{fg}$，其中$\mathcal{V}_{bg}$与$\mathcal{V}_{fg}$分别表示背景实体与前景组件节点集合。边集合划分为三类关系$\mathcal{E} = \mathcal{R}_{BB} \cup \mathcal{R}_{BF} \cup \mathcal{R}_{FF}$，分别对应背景-背景、背景-前景及前景-前景关系，关系类型集合记为$\mathcal{R} = \{\mathcal{R}_{BB}, \mathcal{R}_{BF}, \mathcal{R}_{FF}\}$。

在节点特征初始化阶段，采用并行编码融合多模态属性。对于任意节点$v_i \in \mathcal{V}$，其初始特征表示为$h_i^{(0)} = [\,e_i^{\mathrm{text}} \,\Vert\, e_i^{\mathrm{pos}} \,\Vert\, e_i^{\mathrm{cls}}\,]$，其中$e_i^{\mathrm{text}}$为预训练文本编码器得到的语义表示，$e_i^{\mathrm{pos}}$为绝对位置嵌入，$e_i^{\mathrm{cls}}$为类别嵌入，$\Vert$表示特征拼接。边特征方面，通过关系编码器将离散相对位置标签$r_{ij}$映射为连续关系特征$e_{ij}^{\mathrm{rel}}$，用于刻画节点$v_i$与$v_j$间空间关系，并作为图神经网络消息传递的结构先验。

在完成约束图构建及节点、边特征初始化后，进一步通过图结构推理建模元素之间更复杂的空间关系。为此，本文引入关系感知图神经网络 RGCN 作为核心模块，通过多层消息传递在图中传播信息。经过$L$层推理后，得到最终节点表示
\begin{equation}
\mathbf{H}_{\mathrm{topo}} =
\{h_1^{(L)}, h_2^{(L)}, \dots, h_N^{(L)}\}
\end{equation}
该表示融合节点语义及其拓扑上下文，并作为结构化条件输入后续布局生成模型。

为使节点表示具备明确几何语义，仅依赖无监督消息传递往往不足以充分保留原始约束图的空间关系。因此，在图编码阶段引入辅助监督，通过额外预测任务约束节点表示，从而在特征空间中显式编码拓扑关系与位置信息。具体包括关系重构与位置预测两个任务。

\subsubsection{关系重构损失}

为保留拓扑关系信息，设计关系重构任务作为辅助监督。具体地，构建关系分类器，以节点对特征$(h_i^{(L)}, h_j^{(L)})$为输入，预测其相对位置类别。优化目标为
\begin{equation}
\mathcal{L}_{\mathrm{rel}}
=
-\frac{1}{|\mathcal{E}|}
\sum_{(i,j) \in \mathcal{E}}
\sum_{c=1}^{N_{\mathrm{rel}}}
y_{ij,c}\log(\hat{y}_{ij,c})
\label{eq:rel_loss}
\end{equation}
其中$y_{ij,c}$为真实关系标签的One-hot编码，$\hat{y}_{ij,c}$为预测概率。

\subsubsection{网格位置分类损失}

为增强节点特征的绝对空间感知能力，引入网格位置预测任务。将画布离散为$S \times S$网格，并利用位置解码器根据$h_i^{(L)}$预测节点所属网格索引，其损失为
\begin{equation}
\mathcal{L}_{\mathrm{pos}}
=
-\frac{1}{|\mathcal{V}|}
\sum_{i \in \mathcal{V}}
\sum_{k=1}^{S^2}
\mathbb{I}(t_i = k)\log(\hat{p}_{i,k})
\label{eq:pos_loss}
\end{equation}
其中$\mathbb{I}(\cdot)$为指示函数，$\hat{p}_{i,k}$为节点$v_i$属于第$k$个网格的预测概率。

在图编码阶段，关系重构与位置预测共同作为辅助监督，从相对关系与绝对位置两方面约束节点表示。综合两项损失，整体优化目标为
\begin{equation}
\mathcal{L}_{\mathrm{enc}}
=
\lambda_{\mathrm{rel}}\mathcal{L}_{\mathrm{rel}}
+
\lambda_{\mathrm{pos}}\mathcal{L}_{\mathrm{pos}}
\end{equation}
其中$\lambda_{\mathrm{rel}}$与$\lambda_{\mathrm{pos}}$为权重系数。通过联合优化，节点表示$\mathbf{H}_{\mathrm{topo}}$能够同时编码拓扑关系与空间语义，从而获得更强几何表达能力，并为后续扩散模型提供稳定有效的条件输入。



\xsubsection{基于扩散的布局生成过程}{Diffusion-Based Layout Generation Process}

% 承上启下
% 解释传统的方式的局限
% 引出为啥使用扩散模型
在布局生成任务中，语义约束通常以相对关系的形式给出，例如“位于上方”“居中对齐”等。这类约束往往难以通过一次性预测直接满足，而更适合通过逐步调整的方式在空间中逐渐实现。与此同时，背景视觉先验所提供的空间结构信息也需要在布局生成过程中不断被感知与利用，从而对元素位置进行动态修正。基于上述特点，本文引入条件扩散生成框架，将布局生成过程建模为由粗到细的逐步优化过程。在该框架下，模型从随机初始状态出发，在多模态条件的约束下逐步更新布局结构，使元素之间的空间关系与几何约束能够在生成过程中逐步得到满足，从而获得结构合理且与背景协调的布局结果。


% 面向几何约束的条件扩散布局生成模型 -> 建模

在前述语义解析与背景初始化的基础上，本文进一步考虑整体布局生成问题。在给定背景视觉先验 $\mathbf{I}_{\mathrm{bg}}$ 以及多模态条件集合 $\mathcal{C}$ 的约束下，目标是生成一组满足语义关系与几何约束的完整布局结果，其中前景与背景元素在统一生成过程中进行协同调整，以保证整体结构的一致性与协调性。

具体而言，多模态条件 $\mathcal{C}$ 包含：（1）由语义解析模块得到的元素类别信息及文本语义表示；（2）由结构化约束提取得到的空间拓扑关系集合 $\mathcal{R}$；（3）由背景图像提取的视觉特征表征。设整体布局由归一化边界框参数表示为 $\mathbf{z}_0 = \{(x_i, y_i, w_i, h_i)\}_{i=1}^{N}$，其中包含前景与背景元素的几何参数，则布局生成过程可视为在条件 $\mathcal{C}$ 约束下对 $\mathbf{z}_0$ 的逐步生成过程。

在前向扩散阶段，本文构建具有预定义噪声调度的随机扰动过程，对布局变量逐步注入高斯噪声。在保持背景元素尺度参数不变的前提下，仅对其位置参数与其余布局变量进行扰动，从而在保留视觉结构稳定性的同时引入必要的空间调整自由度。设扩散步数为 $T$，噪声调度序列为 $\{\beta_t\}_{t=1}^{T}$，其中 $0 < \beta_t < 1$，记
$
\alpha_t = 1 - \beta_t, \quad
\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s
$
前向过程定义为：
\begin{equation}
q(\mathbf{z}_t \mid \mathbf{z}_{t-1}) 
= 
\mathcal{N}
\left(
\sqrt{\alpha_t}\,\mathbf{z}_{t-1},
\beta_t \mathbf{I}
\right)
\end{equation}

由此可得：
\begin{equation}
\mathbf{z}_t 
=
\sqrt{\bar{\alpha}_t}\,\mathbf{z}_0
+
\sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon},
\quad
\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
\end{equation}

随着时间步 $t$ 的增加，原始布局结构逐渐被扰动，为后续从随机状态逐步恢复结构提供起点。

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{融合视觉先验与拓扑约束的布局生成框架_v4.png}
  \caption{融合视觉先验与拓扑约束的布局生成框架图}
  \label{fig:layout_generation_framework}
\end{figure}

在反向生成阶段，模型从随机初始化出发，通过逐步更新恢复满足语义约束与空间关系的布局结果。在该过程中，布局的更新不仅依赖当前状态 $\mathbf{z}_t$，还受到多模态条件 $\mathcal{C}$ 的持续约束。为此，引入去噪网络 $\Phi_\theta(\mathbf{z}_t, t, \mathcal{C})$，在每一步中对布局进行修正，使其逐步趋向满足元素之间的相对位置关系与几何一致性。

如图 \ref{fig:layout_generation_framework} 融合视觉先验与拓扑约束的布局生成框架图所示，为使生成过程能够充分利用语义信息与空间关系，本文在去噪网络中引入背景视觉先验编码模块、空间语义编码模块、几何关系感知模块以及布局编码与布局解码模块，通过多模块协同建模元素语义信息与空间拓扑关系，从而在逐步去噪的过程中持续融合语义约束与几何结构信息。通过上述过程，实现从随机初始化到结构合理布局的逐步生成。接下来将分别对各组成模块的结构与实现进行详细介绍。

\subsubsection{背景视觉先验编码}

在海报布局生成任务中，前景元素的空间排布不仅受到语义约束的影响，
还与背景图像中的视觉结构密切相关。例如，背景中的视觉中心、
纹理分布以及空间层次往往会对文本与图形元素的布局位置产生重要影响。
因此，在生成前景布局时引入背景视觉信息，有助于模型理解图像中的
视觉上下文，从而生成与背景内容更加协调的布局结果。基于上述考虑，本文在布局生成框架中引入背景视觉先验编码模块，
用于从参考背景图像中提取稳定的视觉特征，并作为后续布局生成模型的
视觉条件输入。

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{视觉编码模块.png}
  \caption{背景视觉先验编码示意图}
  \label{fig:background_visual_prior_encoding}
\end{figure}

如图\ref{fig:background_visual_prior_encoding}所示，对于上文净化后的背景图像 $\tilde{\mathbf{I}}_{\mathrm{bg}}$ ，本文采用自监督预训练的DINO-ViT \cite{oquab2024dinov} 作为视觉特征提取主干网络。
与传统卷积神经网络相比，Vision Transformer 将输入图像划分为固定大小的图像块（patch），并通过 Transformer编码器进行全局建模，因此能够捕获更加丰富的全局视觉语义信息。

设输入图像尺寸为
$\tilde{\mathbf{I}}_{\mathrm{bg}} \in
\mathbb{R}^{H_{\mathrm{img}}\times W_{\mathrm{img}}\times 3}$，
并将其划分为大小为 $P\times P$ 的图像块，则图像块数量为
$N_p=\frac{H_{\mathrm{img}}}{P}\times\frac{W_{\mathrm{img}}}{P}$。
经过 DINO-ViT 编码器后，可得到输出特征序列
\begin{equation}
Z=[z_{\mathrm{cls}},z_1,z_2,\dots,z_{N_p}]
\in\mathbb{R}^{(N_p+1)\times D}
\label{eq:vit_output}
\end{equation}
其中 $z_{\mathrm{cls}}$ 为分类 token，用于表示图像级语义信息；
其余 $Z_{\mathrm{spatial}}=[z_1,z_2,\dots,z_{N_p}]$ 表示对应图像块的空间 token，$D$ 为特征维度。

由于布局生成任务更关注空间结构信息，本文仅保留空间 token
$Z_{\mathrm{spatial}}$，
并按照图像块的原始空间顺序将其重新排列为二维特征图
$F=\mathrm{Reshape}(Z_{\mathrm{spatial}})$，
其中 $F\in\mathbb{R}^{D\times H_f\times W_f}$，
且 $H_f=\frac{H_{\mathrm{img}}}{P}$，
$W_f=\frac{W_{\mathrm{img}}}{P}$。
为了使 ViT 提取的特征能够更好地适配后续布局生成模型，本文引入轻量级
线性适配器（Linear Adapter）对特征维度进行映射：
\begin{equation}
F_{\mathrm{vis}} = W_{\mathrm{proj}}F + b_{\mathrm{proj}}
\label{eq:adapter}
\end{equation}
其中 $W_{\mathrm{proj}}\in\mathbb{R}^{C\times D}$ 为可学习权重矩阵，
$b_{\mathrm{proj}}$ 为偏置项，最终得到视觉特征图
$F_{\mathrm{vis}}\in\mathbb{R}^{C\times H_f\times W_f}$。

在布局生成过程中，不同元素的几何位置通常对应于图像中的特定区域。
为了获取与候选布局区域相关的局部视觉信息，本文进一步利用 RoIAlign \cite{he2017mask}
操作从视觉特征图中提取区域特征：
\begin{equation}
V=\mathrm{RoIAlign}(F_{\mathrm{vis}},\mathbf{x})
\label{eq:roi_align}
\end{equation}
其中 $\mathbf{x}=(x_1,y_1,x_2,y_2)$ 表示候选布局框，
$V\in\mathbb{R}^{C\times H_r\times W_r}$ 为对应区域的视觉特征表示。

\subsubsection{几何关系感知的空间上下文建模}

在前述模块中，本文分别从语义关系与视觉上下文两个方面对布局元素进行了特征建模。语义约束编码模块通过异构图神经网络对元素之间的语义关系进行推理，获得了包含拓扑结构信息的节点表示；视觉特征提取模块则基于背景图像为每个布局元素生成对应的 RoI 特征，以提供局部视觉信息。然而，仅依赖语义关系表示与局部视觉特征，仍难以充分刻画海报设计中的空间排版逻辑。在实际排版过程中，不同元素之间通常还存在明确的几何关系，例如对齐、相邻、居中和遮挡等。因此，有必要在已有语义表示与视觉特征的基础上进一步引入几何关系建模机制，以增强模型对空间结构的感知能力。


为此，本文设计了几何关系感知模块，通过显式建模布局元素之间的相对几何关系，
对 RoI 特征进行上下文增强。如图\ref{fig:geo_relation_context_modeling}所示，
对于任意两个布局元素 $i$ 和 $j$，
设其边界框表示为 $(x_i,y_i,w_i,h_i)$ 与 $(x_j,y_j,w_j,h_j)$，
其中 $(x,y)$ 表示中心坐标，$w$ 与 $h$ 分别表示宽度和高度。
参考目标检测领域常用的关系建模方式，
本文构建具有尺度不变性的相对几何关系向量：
\begin{equation}
\mathbf{g}_{ij} =
\left[
\log\frac{|x_i-x_j|}{w_j},
\log\frac{|y_i-y_j|}{h_j},
\log\frac{w_i}{w_j},
\log\frac{h_i}{h_j}
\right]
\label{eq:rel_geo}
\end{equation}
该表示能够同时刻画元素之间的相对位置与尺度关系，
并通过对数变换增强尺度变化下的稳定性。
为了提升几何关系的表达能力，
本文利用正弦–余弦位置编码函数 $\mathrm{PE}(\cdot)$
将低维几何关系向量 $\mathbf{g}_{ij}$ 映射到高维嵌入空间，
得到几何关系表示
$\mathbf{R}_{ij}^{p}=\mathrm{PE}(\mathbf{g}_{ij})$。
该嵌入能够保留高频空间结构信息，
从而更好地刻画远距离元素之间的几何依赖。

\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\textwidth]{几何增强的视觉特征v2.png}
  \caption{几何关系感知的空间上下文建模示意图}
  \label{fig:geo_relation_context_modeling}
\end{figure}

在此基础上，本文构建基于几何亲和力的关系注意力机制。
具体而言，通过多层感知机 $\mathrm{MLP}(\cdot)$
对几何嵌入进行变换以获得元素对之间的关系得分，
并通过 $\mathrm{Softmax}$ 函数进行归一化，
从而得到几何注意力权重：
\begin{equation}
\alpha_{ij}
=
\frac{
\exp\big(\mathrm{MLP}(\mathbf{R}_{ij}^{p})\big)
}{
\sum_{k}\exp\big(\mathrm{MLP}(\mathbf{R}_{ik}^{p})\big)
}
\label{eq:geo_attention}
\end{equation}
其中 $\alpha_{ij}$ 表示元素 $i$ 与元素 $j$ 之间的几何关联强度。

最后，利用该几何注意力对所有元素的视觉特征进行加权聚合，
从而获得融合空间上下文信息的特征表示：
\begin{equation}
\mathbf{h}_{i}^{\mathrm{geo}}
=
\sum_{j}\alpha_{ij}\,P(\mathbf{v}_{j})
\label{eq:geo_aggregation}
\end{equation}
其中 $\mathbf{v}_{j}$ 表示第 $j$ 个元素的 RoI 视觉特征，
$P(\cdot)$ 为线性映射函数。
通过该聚合过程，每个元素的特征不仅包含自身的视觉信息，
还能够融合与其具有显著几何关联的其他元素特征，
从而显式建模布局中的空间依赖结构。
最终得到的几何增强特征将作为后续布局生成模块的重要输入，
以提升生成布局与整体空间结构之间的协调性。

\subsubsection{布局表示与多模态解码}

在上一阶段中，语义关系编码模块与几何关系建模模块已经分别从拓扑结构与空间几何两个角度，
为每个前景元素构建了上下文增强的关系表征。其中，RGCN 模块获得的
$\mathbf{H}_{\mathrm{topo}}$ 描述了元素之间的语义依赖与拓扑结构关系，
而几何关系感知模块得到的 $\mathbf{H}_{\mathrm{geo}}$ 则刻画了元素之间的相对空间布局。
此外，视觉特征提取模块还提供了背景图像的视觉上下文特征 $F_{\mathrm{vis}}$。
然而，上述多模态信息仍然以图结构或区域特征的形式存在，
尚不能直接作为扩散去噪网络的条件输入。
因此，需要进一步构建统一的布局表示，
并在生成过程中实现不同模态特征之间的协同交互。
为此，本文设计了一个布局生成模块，
首先将区域语义描述与几何信息编码为连续的布局表示，
随后通过多模态注意力机制融合视觉、拓扑与几何条件特征，
从而逐步预测前景元素的几何布局。

\paragraph{布局表示}

如图\ref{fig:layout_generation_framework}中布局编码器所示，为了将离散的布局约束转化为可供去噪网络处理的连续条件表示，
本文首先为每个前景元素构建统一的布局 Token。
对于第 $i$ 个前景元素，
其输入由区域描述文本 $c_i$ 与归一化边界框坐标
$b_i=(x_i,y_i,w_i,h_i)$ 组成。
参考 GLIGEN \cite{li2023gligen} 的设计思想，
本文采用并行编码策略处理语义与几何两类信息。
首先利用预训练CLIP \cite{schuhmann2022laionb}文本编码器
$\boldsymbol{\tau}(\cdot)$
提取文本语义表示 $\mathbf{t}_i=\boldsymbol{\tau}(c_i)$。
随后对边界框坐标进行几何编码。
由于低维坐标在原始空间中难以表达高频空间变化，
本文采用傅里叶特征映射将坐标投影至高维空间。
其编码形式为
\begin{equation}
\mathrm{Fourier}(b_i)=
[\sin(2\pi \mathbf{B} b_i),\cos(2\pi \mathbf{B} b_i)]
\label{eq:fourier_encoding}
\end{equation}
其中 $\mathbf{B}$ 为固定的高斯随机矩阵。
该映射能够增强模型对细微位置偏移与尺度变化的表达能力。

随后将语义嵌入与几何嵌入在通道维度进行拼接，
并通过多层感知机（MLP）进行特征融合，
得到第 $i$ 个元素的布局表示
$h_i^l=\mathrm{MLP}([\mathbf{t}_i \parallel \mathrm{Fourier}(b_i)])$。
对所有前景元素进行编码后，
得到布局 Token 序列
$\mathbf{H}_l=[h_1^l,\dots,h_N^l]\in\mathbb{R}^{N\times D}$，
其作为后续布局生成过程的主生成流。

\paragraph{多模态解码}

在获得布局 Token 表示后，
本文进一步设计多模态布局解码器，
用于在扩散去噪过程中融合不同来源的条件信息。
如图\ref{fig:layout_generation_framework}布局解码器所示，与传统的单向条件注入方式不同，
该模块借鉴 MM-DiT \cite{pmlr-v235-esser24a-SD3} 的联合注意力机制，
通过统一的注意力计算实现布局表示与条件特征之间的双向交互。
给定布局 Token $\mathbf{H}_l$ 以及任意条件模态特征
$\mathbf{H}_c$，
两者首先通过线性投影映射为
查询、键和值向量，
随后在序列维度上拼接并执行统一的注意力计算：
\begin{equation}
[\mathbf{H}_l',\mathbf{H}_c'] =
\mathrm{Attention}
\big(
[\mathbf{Q}_l,\mathbf{Q}_c],
[\mathbf{K}_l,\mathbf{K}_c],
[\mathbf{V}_l,\mathbf{V}_c]
\big)
\label{eq:mm_attention}
\end{equation}
该机制允许布局 Token 与条件 Token
在同一特征空间中进行信息交换，
从而在生成过程中动态调整各类条件信息的贡献。

在此基础上，
解码器分别引入三种条件特征进行并行交互：
视觉特征交互分支利用背景特征图
$F_{\mathrm{vis}}$，
使生成的布局能够感知背景纹理与视觉结构，
从而避免重要元素落入复杂区域；
拓扑关系交互分支利用
RGCN 输出的节点嵌入
$\mathbf{H}_{\mathrm{topo}}$，
使生成布局满足语义解析阶段得到的空间关系约束；
几何关系交互分支利用
几何关系模块得到的特征
$\mathbf{H}_{\mathrm{geo}}$，
用于强化元素之间的相对距离与尺度比例关系。
经过三种模态特征的交互更新后，
分别得到增强后的布局表示
$\mathbf{H}_l^{\mathrm{vis}}$、
$\mathbf{H}_l^{\mathrm{topo}}$ 与
$\mathbf{H}_l^{\mathrm{geo}}$。
随后在通道维度对其进行融合：
\begin{equation}
\mathbf{H}_{\mathrm{final}}=
\mathrm{Concat}
(\mathbf{H}_l^{\mathrm{vis}},
\mathbf{H}_l^{\mathrm{topo}},
\mathbf{H}_l^{\mathrm{geo}})
\mathbf{W}_{\mathrm{fuse}}
\label{eq:final_fusion}
\end{equation}

最终，
融合后的布局表示被输入至 MLP 预测头，
将高维潜在特征映射为二维边界框坐标，
用于预测当前扩散步的去噪结果。
通过上述多模态协同解码机制，
模型能够在语义逻辑、几何结构与视觉上下文之间建立一致的空间对齐关系，
从而生成既满足设计约束又具有视觉合理性的布局结构。
