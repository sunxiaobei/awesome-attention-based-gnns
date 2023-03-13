# Awesome Attention-based GNNs

- A collection of resources on attention-based graph neural networks.

- Welcome to submit a pull request to add more awesome papers.

```markdown
- [x] [journal] [model] paper_title [[paper]](link) [[code]](link)
```

## Table of Contents
- [Surveys](#Surveys)
- [GRANs](#GRANs) : (Graph Recurrent Attention Networks)
- [GATs](#GATs) : (Graph Attention Networks)
- [Graph Transformers](#GraphTransformers) : (Graph Transformers)

## Survey

- [x] [TKDD2019] [survey] Attention Models in Graphs: A Survey [[paper]](https://doi.org/10.1145/3363574) 

## GRANs

### GRU Attention
- [x] [ICLR2016] [GGNN] Gated Graph Sequence Neural Networks [[paper]](http://arxiv.org/abs/1511.05493) [[code]](https://github.com/yujiali/ggnn)
- [x] [UAI2018] [GaAN] GaAN: Gated Attention Networks for Learning on Large and Spatiotemporal Graphs [[paper]](http://arxiv.org/abs/1803.07294) [[code]](https://github.com/jennyzhang0215/GaAN)
- [x] [ICML2018] [GraphRNN] Graphrnn: Generating realistic graphs with deep auto-regressive models [[paper]](http://arxiv.org/abs/1802.08773) [[code]](https://github.com/snap-stanford/GraphRNN)
- [x] [NeurIPS2019] [GRAN] Efficient Graph Generation with Graph Recurrent Attention Networks [[paper]](http://arxiv.org/abs/1910.00760) [[code]](https://github.com/lrjconan/GRAN)
- [x] [T-SP2020] [GRNN] Gated Graph Recurrent Neural Networks [[paper]](https://arxiv.org/abs/2002.01038) 
- [x] [2021] [GR-GAT] Gated Relational Graph Attention Networks [[paper]](https://openreview.net/forum?id=v-9E8egy_i) 

### LSTM Attention
- [x] [NeurIPS2017] [GraphSage] Inductive representation learning on large graphs [[paper]](https://arxiv.org/abs/1706.02216) [[code]](https://github.com/williamleif/GraphSAGE)
- [x] [ICML2018] [JK-Net] Representation Learning on Graphs with Jumping Knowledge Networks [[paper]](http://proceedings.mlr.press/v80/xu18c.html) 
- [x] [KDD2018] [GAM] Graph Classification using Structural Attention [[paper]](https://doi.org/10.1145/3219819.3219980) [[code]](https://github.com/benedekrozemberczki/GAM)
- [x] [AAAI2019] [GeniePath] GeniePath: Graph Neural Networks with Adaptive Receptive Paths [[paper]](http://arxiv.org/abs/1802.00910) 

## GATs
- [x] [ICLR2018] [GAT] Graph Attention Networks [[paper]](https://openreview.net/forum?id=rJXMpikCZ) [[code]](https://github.com/PetarV-/GAT)
- [x] [NeurIPS2019] [C-GAT] Improving Graph Attention Networks with Large Margin-based Constraints [[paper]](http://arxiv.org/abs/1910.11945) 
- [x] [IJCAI2020] [CPA] Improving Attention Mechanism in Graph Neural Networks via Cardinality Preservation [[paper]](https://pubmed.ncbi.nlm.nih.gov/32782421/) [[code]](https://github.com/zetayue/CPA)
- [x] [ICLR2022] [GATv2] How Attentive are Graph Attention Networks? [[paper]](http://arxiv.org/abs/2105.14491) [[code]](https://github.com/tech-srl/how_attentive_are_gats)
- [x] [ICLR2022] [PPRGAT] Personalized PageRank Meets Graph Attention Networks [[paper]](https://openreview.net/forum?id=XNYOJD0QdBD) 
- [x] [KDD2021] [Simple-HGN] Are we really making much progress? Revisiting, benchmarking and refining heterogeneous graph neural networks [[paper]](https://doi.org/10.1145/3447548.3467350) [[code]](https://github.com/THUDM/HGB)
- [x] [ICLR2021] [SuperGAT] How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision [[paper]](https://openreview.net/forum?id=Wi5KUNlqWty) [[code]](https://github.com/dongkwan-kim/SuperGAT)
- [x] [NeurIPS2021] [DMP] Diverse Message Passing for Attribute with Heterophily [[paper]](https://openreview.net/forum?id=4jPVcKEYpSZ) 
- [x] [KDD2019] [GANet] Graph Representation Learning via Hard and Channel-Wise Attention Networks [[paper]](https://doi.org/10.1145/3292500.3330897) 
- [x] [NeurIPS2021] [CAT] Learning Conjoint Attentions for Graph Neural Nets [[paper]](http://arxiv.org/abs/2102.03147) [[code]](https://github.com/he-tiantian/cats)
- [x] [TransNNLS2021] [MSNA] Neighborhood Attention Networks With Adversarial Learning for Link Prediction [[paper]](https://ieeexplore.ieee.org/document/9174790) 
- [x] [arXiv2018] [GGAT] Deeply learning molecular structure-property relationships using attention- and gate-augmented graph convolutional network [[paper]](http://arxiv.org/abs/1805.10988) 
- [x] [NeurIPS2019] [HGCN] Hyperbolic Graph Convolutional Neural Networks [[paper]](http://arxiv.org/abs/1910.12933) 
- [x] [TBD2021] [HAT] Hyperbolic graph attention network [[paper]](https://arxiv.org/abs/1912.03046) 
- [x] [IJCAI2020] [Hype-HAN] Hype-HAN: Hyperbolic Hierarchical Attention Network for Semantic Embedding [[paper]](https://www.ijcai.org/proceedings/2020/552) 
- [x] [ICLR2019] [] Hyperbolic Attention Networks [[paper]](http://arxiv.org/abs/1805.09786) 
- [x] [ICLR2018] [AGNN] Attention-based Graph Neural Network for Semi-supervised Learning [[paper]](http://arxiv.org/abs/1803.03735) 
- [x] [KDD2018] [HTNE] Embedding Temporal Network via Neighborhood Formation [[paper]](https://doi.org/10.1145/3219819.3220054) [[code]](https://github.com/hongyuanmei/neurawkes)
- [x] [CVPR2020] [DKGAT] Distilling Knowledge from Graph Convolutional Networks [[paper]](http://arxiv.org/abs/2003.10477) [[code]](https://github.com/ihollywhy/DistillGCN.PyTorch)
- [x] [ICCV2021] [DAGL] Dynamic Attentive Graph Learning for Image Restoration [[paper]](http://arxiv.org/abs/2109.06620) [[code]](https://github.com/jianzhangcs/dagl)
- [x] [AAAI2021] [SCGA] Structured Co-reference Graph Attention for Video-grounded Dialogue [[paper]](http://arxiv.org/abs/2103.13361) 
- [x] [AAAI2021] [Co-GAT] Co-GAT: A Co-Interactive Graph Attention Network for Joint Dialog Act Recognition and Sentiment Classification [[paper]](http://arxiv.org/abs/2012.13260) 
- [x] [ACL2020] [ED-GAT] Entity-Aware Dependency-Based Deep Graph Attention Network for Comparative Preference Classification [[paper]](https://aclanthology.org/2020.acl-main.512) 
- [x] [EMNLP2019] [TD-GAT] Syntax-Aware Aspect Level Sentiment Classification with Graph Attention Networks [[paper]](http://arxiv.org/abs/1909.02606) 
- [x] [WWW2020] [GATON] Graph Attention Topic Modeling Network [[paper]](https://doi.org/10.1145/3366423.3380102) 
- [x] [KDD2017] [GRAM] GRAM: Graph-based attention model for healthcare representation learning [[paper]](https://doi.org/10.1145/3097983.3098126) [[code]](https://github.com/mp2893/gram)

- [x] [IJCAI2019] [SPAGAN] SPAGAN: Shortest Path Graph Attention Network [[paper]](https://www.ijcai.org/proceedings/2019/569) 
- [x] [PKDD2021] [PaGNN] Inductive Link Prediction with Interactive Structure Learning on Attributed Graph [[paper]](https://link.springer.com/book/10.1007/978-3-030-86486-6) 
- [x] [arXiv2019] [DeepLinker] Link Prediction via Graph Attention Network [[paper]](http://arxiv.org/abs/1910.04807) 
- [x] [IJCNN2020] [CGAT] Heterogeneous Information Network Embedding with Convolutional Graph Attention Networks [[paper]](https://ieeexplore.ieee.org/document/9206610) 
- [x] [ICLR2020] [ADSF] Adaptive Structural Fingerprints for Graph Attention Networks [[paper]](https://openreview.net/forum?id=BJxWx0NYPr) 
- [x] [KDD2021] [T-GAP] Learning to Walk across Time for Interpretable Temporal Knowledge Graph Completion [[paper]](https://doi.org/10.1145/3447548.3467292) [[code]](https://github.com/sharkmir1/T-GAP)
- [x] [NeurIPS2018] [MAF] Modeling Attention Flow on Graphs [[paper]](http://arxiv.org/abs/1811.00497) 
- [x] [IJCAI2021] [MAGNA] Multi-hop Attention Graph Neural Network [[paper]](http://arxiv.org/abs/2009.14332) [[code]](https://github.com/xjtuwgt/gnn-magna)


- [x] [AAAI2020] [SNEA] Learning Signed Network Embedding via Graph Attention [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/5911) 
- [x] [ICANN2019] [SiGAT] Signed Graph Attention Networks [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-30493-5_53) 
- [x] [ICLR2019] [RGAT] Relational Graph Attention Networks [[paper]](http://arxiv.org/abs/1904.05811) [[code]](https://github.com/babylonhealth/rgat)
- [x] [arXiv2018] [EAGCN] Edge attention-based multi-relational graph convolutional networks [[paper]](http://arxiv.org/abs/1802.04944) [[code]](https://github.com/Luckick/EAGCN)
- [x] [KDD2021] [WRGNN] Breaking the Limit of Graph Neural Networks by Improving the Assortativity of Graphs with Local Mixing Patterns [[paper]](https://doi.org/10.1145/3447548.3467373) [[code]](https://github.com/susheels/gnns-and-local-assortativity)
- [x] [AAAI2020] [HetSANN] An Attention-based Graph Neural Network for Heterogeneous Structural Learning [[paper]](http://arxiv.org/abs/1912.10832) [[code]](https://github.com/didi/hetsann)
- [x] [AAAI2020] [TALP] Type-Aware Anchor Link Prediction across Heterogeneous Networks Based on Graph Attention Network [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/5345) 
- [x] [KDD2019] [KGAT] KGAT: Knowledge Graph Attention Network for Recommendation [[paper]](http://arxiv.org/abs/1905.07854) [[code]](https://github.com/xiangwang1223/knowledge_graph_attention_network)
- [x] [KDD2019] [GATNE] Representation Learning for Attributed Multiplex Heterogeneous Network [[paper]](https://doi.org/10.1145/3292500.3330964) [[code]](https://github.com/THUDM/GATNE)
- [x] [AAAI2021] [RelGNN] Relation-aware Graph Attention Model With Adaptive Self-adversarial Training [[paper]](http://arxiv.org/abs/2102.07186) 
- [x] [KDD2020] [CGAT] Graph Attention Networks over Edge Content-Based Channels [[paper]](https://doi.org/10.1145/3394486.3403233) 
- [x] [ACL2019] [AFE] Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs [[paper]](http://arxiv.org/abs/1906.01195) [[code]](https://github.com/deepakn97/relationPrediction)
- [x] [CIKM2021] [DisenKGAT] DisenKGAT: Knowledge Graph Embedding with Disentangled Graph Attention Network [[paper]](https://arxiv.org/abs/2108.09628v2) 
- [x] [AAAI2021] [GTAN] Graph-Based Tri-Attention Network for Answer Ranking in CQA [[paper]](http://arxiv.org/abs/2103.03583) 
- [x] [ACL2020] [R-GAT] Relational Graph Attention Network for Aspect-based Sentiment Analysis [[paper]](http://arxiv.org/abs/2004.12362) [[code]](https://github.com/shenwzh3/RGAT-ABSA)
- [x] [ICCV2019] [ReGAT] Relation-Aware Graph Attention Network for Visual Question Answering [[paper]](https://arxiv.org/abs/1903.12314v3) [[code]](https://github.com/linjieli222/VQA_ReGAT)
- [x] [AAAI2021] [AD-GAT] Modeling the Momentum Spillover Effect for Stock Prediction via Attribute-Driven Graph Attention Networks [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/16077) 


- [x] [NeurIPS2018] [GAW] Watch your step: learning node embeddings via graph attention [[paper]](https://arxiv.org/abs/1710.09599v2) 
- [x] [TPAMI2021] [NLGAT] Non-Local Graph Neural Networks [[paper]](https://arxiv.org/abs/2005.14612) 
- [x] [NeurIPS2019] [ChebyGIN] Understanding Attention and Generalization in Graph Neural Networks [[paper]](http://papers.nips.cc/paper/8673-understanding-attention-and-generalization-in-graph-neural-networks.pdf) [[code]](https://github.com/bknyaz/graph_attention_pool)
- [x] [ICML2019] [SAGPool] Self-Attention Graph Pooling [[paper]](http://proceedings.mlr.press/v97/lee19c.html) [[code]](https://github.com/inyeoplee77/SAGPool)
- [x] [ICCV2019] [Attpool] Attpool: Towards hierarchical feature representation in graph convolutional networks via attention mechanism [[paper]](https://ieeexplore.ieee.org/document/9009471) 


- [x] [WWW2019] [HAN] Heterogeneous Graph Attention Network [[paper]](https://doi.org/10.1145/3308558.3313562) [[code]]( https://github.com/Jhy1993/HAN)
- [x] [NC2022] [PSHGAN] Heterogeneous graph embedding by aggregating meta-path and meta-structure through attention mechanism [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S092523122101479X) 
- [x] [IJCAI2017] [PRML] Link prediction via ranking metric dual-level attention network learning [[paper]](https://www.ijcai.org/proceedings/2017/493) 
- [x] [WSDM2022] [GraphHAM] Graph Embedding with Hierarchical Attentive Membership [[paper]](http://arxiv.org/abs/2111.00604) 
- [x] [AAAI2020] [RGHAT] Relational Graph Neural Network with Hierarchical Attention for Knowledge Graph Completion [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/6508) 
- [x] [AAAI2019] [LAN] Logic Attention Based Neighborhood Aggregation for Inductive Knowledge Graph Embedding [[paper]](https://doi.org/10.1609/aaai.v33i01.33017152) [[code]](https://github.com/wangpf3/LAN)
- [x] [2022] [EFEGAT] Learning to Solve an Order Fulfillment Problem in Milliseconds with Edge-Feature-Embedded Graph Attention [[paper]](https://openreview.net/forum?id=qPQRIj_Y_EW) 
- [x] [WWW2019] [DANSER] Dual Graph Attention Networks for Deep Latent Representation of Multifaceted Social Effects in Recommender Systems [[paper]](https://doi.org/10.1145/3308558.3313442) [[code]](https://github.com/echo740/DANSER-WWW-19)
- [x] [WWW2019] [UVCAN] User-Video Co-Attention Network for Personalized Micro-video Recommendation [[paper]](https://doi.org/10.1145/3308558.3313513) 
- [x] [KBS2020] [HAGERec] HAGERec: Hierarchical Attention Graph Convolutional Network Incorporating Knowledge Graph for Explainable Recommendation [[paper]](https://www.sciencedirect.com/science/article/pii/S0950705120304196) 
- [x] [Bio2021] [GCATSL] Graph contextualized attention network for predicting synthetic lethality in human cancers [[paper]](https://doi.org/10.1093/bioinformatics/btab110) [[code]](https://github.com/longyahui/GCATSL)
- [x] [ICMR2020] [DAGC] DAGC: Employing Dual Attention and Graph Convolution for Point Cloud based Place Recognition [[paper]](https://doi.org/10.1145/3372278.3390693) [[code]](https://github.com/dawenzi123/DAGCN)
- [x] [AAAI2020] [AGCN] Graph Attention Based Proposal 3D ConvNets for Action Detection [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/5893) 
- [x] [EMNLP2019] [HGAT] HGAT: Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification [[paper]](https://dl.acm.org/doi/10.1145/3450352) [[code]](https://github.com/ytc272098215/HGAT)


- [x] [ICLR2020] [Hyper-SAGNN] Hyper-Sagnn: A Self-Attention based Graph Neural Network for Hypergraphs [[paper]](http://arxiv.org/abs/1911.02613) [[code]](https://github.com/ma-compbio/Hyper-SAGNN)
- [x] [CIKM2021] [HHGR] Double-Scale Self-Supervised Hypergraph Learning for Group Recommendation [[paper]](http://arxiv.org/abs/2109.04200) [[code]](https://github.com/0411tony/hhgr)
- [x] [ICDM2021] [HyperTeNet] HyperTeNet: Hypergraph and Transformer-based Neural Network for Personalized List Continuation [[paper]](http://arxiv.org/abs/2110.01467) [[code]](https://github.com/mvijaikumar/hypertenet)
- [x] [PR2020] [Hyper-GAT] Hypergraph Convolution and Hypergraph Attention [[paper]](http://arxiv.org/abs/1901.08150) [[code]](https://github.com/rusty1s/pytorch_geometric)
- [x] [CVPR2020] [] Hypergraph attention networks for multimodal learning [[paper]](https://ieeexplore.ieee.org/document/9157723) 


- [x] [KDD2020] [DAGNN] Towards Deeper Graph Neural Networks [[paper]](https://doi.org/10.1145/3394486.3403076) [[code]](https://github.com/divelab/DeeperGNN)
- [x] [T-NNLS2020] [AP-GCN]  Adaptive propagation graph convolutional network [[paper]](https://arxiv.org/abs/2002.10306) [[code]](https://github.com/spindro/AP-GCN)
- [x] [CIKM2021] [TDGNN] Tree Decomposed Graph Neural Network [[paper]](https://doi.org/10.1145/3459637.3482487) [[code]](https://github.com/YuWVandy/TDGNN)
- [x] [arXiv2020] [GMLP] GMLP: Building Scalable and Flexible Graph Neural Networks with Feature-Message Passing [[paper]](http://arxiv.org/abs/2104.09880) 
- [x] [KDD2022] [GAMLP] Graph Attention Multi-layer Perceptron [[paper]](http://arxiv.org/abs/2108.10097) [[code]](https://github.com/zwt233/GAMLP)


- [x] [AAAI2021] [FAGCN] Beyond Low-frequency Information in Graph Convolutional Networks [[paper]](http://arxiv.org/abs/2101.00797) [[code]](https://github.com/bdy9527/FAGCN)
- [x] [arXiv2021] [ACM] Is Heterophily A Real Nightmare For Graph Neural Networks To Do Node Classification? [[paper]](http://arxiv.org/abs/2109.05641) 


- [x] [KDD2020] [AM-GCN] AM-GCN: Adaptive Multi-channel Graph Convolutional Networks [[paper]](https://doi.org/10.1145/3394486.3403177) [[code]](https://github.com/zhumeiqiBUPT/AM-GCN)
- [x] [AAAI2021] [UAG] Uncertainty-aware Attention Graph Neural Network for Defending Adversarial Attacks [[paper]](http://arxiv.org/abs/2009.10235) 
- [x] [CIKM2021] [MV-GNN] Semi-Supervised and Self-Supervised Classification with Multi-View Graph Neural Networks [[paper]](https://doi.org/10.1145/3459637.3482477) 
- [x] [KSEM2021] [GENet] Graph Ensemble Networks for Semi-supervised Embedding Learning [[paper]](https://dl.acm.org/doi/abs/10.1007/978-3-030-82136-4_33) 
- [x] [NN2020] [MGAT] MGAT: Multi-view Graph Attention Networks [[paper]](https://www.sciencedirect.com/science/article/pii/S0893608020303105) 
- [x] [CIKM2017] [MVE] An Attention-based Collaboration Framework for Multi-View Network Representation Learning [[paper]](https://doi.org/10.1145/3132847.3133021) 
- [x] [NC2021] [EAGCN] Multi-view spectral graph convolution with consistent edge attention for molecular modeling [[paper]](https://www.sciencedirect.com/science/article/pii/S092523122100271X) 
- [x] [SIGIR2020] [GCE-GNN] Global Context Enhanced Graph Neural Networks for Session-based Recommendation [[paper]](http://arxiv.org/abs/2106.05081) 


- [x] [ICLR2019] [DySAT] Dynamic Graph Representation Learning via Self-Attention Networks [[paper]](https://arxiv.org/abs/1812.09430v2) [[code]](https://github.com/aravindsankar28/DySAT)
- [x] [PAKDD2020] [TemporalGAT] TemporalGAT: Attention-Based Dynamic Graph Representation Learning [[paper]](https://dl.acm.org/doi/10.1007/978-3-030-47426-3_32) 
- [x] [IJCAI2021] [GAEN] GAEN: Graph Attention Evolving Networks [[paper]](https://www.ijcai.org/proceedings/2021/213) [[code]](https://github.com/codeshareabc/GAEN)
- [x] [CIKM2019] [MMDNE] Temporal Network Embedding with Micro- and Macro-dynamics [[paper]](https://doi.org/10.1145/3357384.3357943) [[code]](https://github.com/rootlu/MMDNE)
- [x] [ICLR2020] [TGAT] Inductive Representation Learning on Temporal Graphs [[paper]](http://arxiv.org/abs/2002.07962) [[code]](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs)
- [x] [2022] [TR-GAT] Time-Aware Relational Graph Attention Network for Temporal Knowledge Graph Embeddings [[paper]](https://openreview.net/forum?id=ShtJLsF7cbb) 
- [x] [CVPR2022] [T-GNN] Adaptive Trajectory Prediction via Transferable GNN [[paper]](http://arxiv.org/abs/2203.05046) 
- [x] [AAAI2018] [ST-GCN] Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17135) [[code]](https://github.com/yysijie/st-gcn)
- [x] [AAAI2020] [GMAN] GMAN: A Graph Multi-Attention Network for Traffic Prediction [[paper]](http://arxiv.org/abs/1911.08415) 
- [x] [AAAI2019] [ASTGCN] Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/3881) [[code]](https://github.com/Davidham3/ASTGCN)
- [x] [KDD2020] [ConSTGAT] ConSTGAT: Contextual Spatial-Temporal Graph Attention Network for Travel Time Estimation at Baidu Maps [[paper]](https://doi.org/10.1145/3394486.3403320) 


- [x] [ICLR2022] [RainDrop] Graph-Guided Network for Irregularly Sampled Multivariate Time Series [[paper]](http://arxiv.org/abs/2110.05357) 
- [x] [ICLR2021] [mTAND] Multi-Time Attention Networks for Irregularly Sampled Time Series [[paper]](http://arxiv.org/abs/2101.10318) [[code]](https://github.com/reml-lab/mTAN)
- [x] [ICDM2020] [MTAD-GAT] Multivariate Time-series Anomaly Detection via Graph Attention Network [[paper]](http://arxiv.org/abs/2009.02040) 
- [x] [WWW2020] [GACNN] Towards Fine-grained Flow Forecasting: A Graph Attention Approach for Bike Sharing Systems [[paper]](https://dl.acm.org/doi/10.1145/3366423.3380097) 
- [x] [IJCAI2018] [GeoMAN] GeoMAN: Multi-level Attention Networks for Geo-sensory Time Series Prediction [[paper]](https://www.ijcai.org/proceedings/2018/476) 



## GraphTransformers

- [x] [arXiv2019] [GTR] Graph transformer [[paper]](https://openreview.net/forum?id=HJei-2RcK7) 
- [x] [arXiv2020] [U2GNN] Universal Self-Attention Network for Graph Classification [[paper]](http://arxiv.org/abs/1909.11855) [[code]](https://github.com/daiquocnguyen/Graph-Transformer)
- [x] [WWW2022] [UGformer] Universal Graph Transformer Self-Attention Networks [[paper]](http://arxiv.org/abs/1909.11855) [[code]](https://github.com/daiquocnguyen/Graph-Transformer)
- [x] [ICLR2021] [GMT] Accurate learning of graph representations with graph multiset pooling [[paper]](http://arxiv.org/abs/2102.11533) [[code]](https://github.com/JinheonBaek/GMT)
- [x] [KDDCup2021] [Graphormer] Do Transformers Really Perform Bad for Graph Representation? [[paper]](http://arxiv.org/abs/2106.08279) [[code]](https://github.com/microsoft/Graphormer)
- [x] [NeurIPS2021] [Graphormer] Do Transformers Really Perform Bad for Graph Representation? [[paper]](http://arxiv.org/abs/2106.05234) [[code]](https://github.com/microsoft/Graphormer)
- [x] [NeurIPS2021] [HOT] Transformers Generalize DeepSets and Can be Extended to Graphs and Hypergraphs [[paper]](http://arxiv.org/abs/2110.14416) [[code]](https://github.com/jw9730/hot)
- [x] [NeurIPS2020] [GROVER] Self-Supervised Graph Transformer on Large-Scale Molecular Data [[paper]](http://arxiv.org/abs/2007.02835) [[code]](https://github.com/tencent-ailab/grover)
- [x] [ICML(Workshop)2019] [PAGAT] Path-augmented graph transformer network [[paper]](https://arxiv.org/abs/1905.12712) [[code]](https://github.com/benatorc/PA-Graph-Transformer)
- [x] [AAAI2021] [GTA] GTA: Graph Truncated Attention for Retrosynthesis [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/16131) 
- [x] [AAAI2021] [GT] A generalization of transformer networks to graphs [[paper]](http://arxiv.org/abs/2012.09699) [[code]](https://github.com/graphdeeplearning/graphtransformer)
- [x] [NeurIPS2021] [SAN] Rethinking graph transformers with spectral attention [[paper]](http://arxiv.org/abs/2106.03893) [[code]](https://github.com/DevinKreuzer/SAN)
- [x] [2020] [GraphBert] Graph-bert: Only attention is needed for learning graph representations [[paper]](http://arxiv.org/abs/2001.05140) [[code]](https://github.com/jwzhanggy/Graph-Bert)
- [x] [ICML2021] [] Lipschitz Normalization for Self-Attention Layers with Application to Graph Neural Networks [[paper]](http://arxiv.org/abs/2103.04886) 
- [x] [IJCAI2021] [UniMP] Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification [[paper]](http://arxiv.org/abs/2009.03509) [[code]](https://github.com/PaddlePaddle/PGL/tree/main/ogb_examples/nodeproppred/unimp)
- [x] [NeurIPS2019] [GTN] Graph Transformer Networks [[paper]](http://arxiv.org/abs/1911.06455) [[code]](https://github.com/seongjunyun/Graph_Transformer_Networks)
- [x] [KDD2020] [TagGen] A Data-Driven Graph Generative Model for Temporal Interaction Networks [[paper]](https://doi.org/10.1145/3394486.3403082) 
- [x] [NeurIPS2021] [GraphFormers] GraphFormers: GNN-nested Transformers for Representation Learning on Textual Graph [[paper]](http://arxiv.org/abs/2105.02605) 
- [x] [WWW2020] [HGT] Heterogeneous Graph Transformer [[paper]](https://doi.org/10.1145/3366423.3380027) 
- [x] [AAAI2020] [GTOS] Graph transformer for graph-to-sequence learning [[paper]](http://arxiv.org/abs/1911.07470) [[code]](https://github.com/jcyk/gtos)
- [x] [NAACL2019] [GraphWriter] Text Generation from Knowledge Graphs with Graph Transformers [[paper]](http://arxiv.org/abs/1904.02342) [[code]](https://github.com/rikdz/GraphWriter)
- [x] [AAAI2021] [KHGT] Knowledge-Enhanced Hierarchical Graph Transformer Network for Multi-Behavior Recommendation [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/16576) [[code]](https://github.com/akaxlh/KHGT)
- [x] [AAAI2021] [GATE] GATE: Graph Attention Transformer Encoder for Cross-lingual Relation and Event Extraction [[paper]](http://arxiv.org/abs/2010.03009) 
- [x] [NeurIPS2021] [STAGIN] Learning Dynamic Graph Representation of Brain Connectome with Spatio-Temporal Attention [[paper]](http://arxiv.org/abs/2105.13495) [[code]](https://github.com/egyptdj/stagin)

## Next

- [x] [journal] [model] paper_title [[paper]](link) [[code]](link)

## Citation
