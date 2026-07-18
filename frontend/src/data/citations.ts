import type { Citation } from '@fasl-work/caos-app-shell';

// The reference spine, transcribed from the persisted research dossiers (wip/fisura/research/,
// verified against primary sources 2026-07-18). Grows as each unit lands. Identifiers (DOI/arXiv)
// appear ONLY where the dossier verified them; entries without one cite venue and year as recorded.
// Inline <Cite id="..." /> resolves against this list via the CitationsProvider at the app root.
export const CITATIONS: Citation[] = [
  // --- classical era ---
  {
    id: 'shi2016crackforest',
    label: 'Shi et al. 2016',
    citation:
      'Shi Y., Cui L., Qi Z., Meng F., Chen Z. (2016). Automatic Road Crack Detection Using Random Structured Forests. IEEE Transactions on Intelligent Transportation Systems 17(12).',
    url: 'https://github.com/cuilimeng/CrackForest-dataset',
  },
  {
    id: 'zou2012cracktree',
    label: 'Zou et al. 2012',
    citation:
      'Zou Q., Cao Y., Li Q., Mao Q., Wang S. (2012). CrackTree: Automatic crack detection from pavement images. Pattern Recognition Letters 33(3).',
  },
  {
    id: 'amhaz2016mps',
    label: 'Amhaz et al. 2016',
    citation:
      'Amhaz R., Chambon S., Idier J., Baltazart V. (2016). Automatic Crack Detection on Two-Dimensional Pavement Images: An Algorithm Based on Minimal Path Selection. IEEE Transactions on Intelligent Transportation Systems 17(10).',
  },
  // --- learned era ---
  {
    id: 'zou2019deepcrack',
    label: 'Zou et al. 2019',
    citation:
      'Zou Q., Zhang Z., Li Q., Qi X., Wang Q., Wang S. (2019). DeepCrack: Learning Hierarchical Convolutional Features for Crack Detection. IEEE Transactions on Image Processing 28(3), 1498-1512.',
    doi: '10.1109/TIP.2018.2878966',
  },
  {
    id: 'liu2019deepcrack',
    label: 'Liu et al. 2019',
    citation:
      'Liu Y., Yao J., Lu X., Xie R., Li L. (2019). DeepCrack: A deep hierarchical feature learning architecture for crack segmentation. Neurocomputing 338, 139-153.',
    doi: '10.1016/j.neucom.2019.01.036',
  },
  {
    id: 'yang2019fphbn',
    label: 'Yang et al. 2019',
    citation:
      'Yang F., Zhang L., Yu S., Prokhorov D., Mei X., Ling H. (2019). Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection. IEEE Transactions on Intelligent Transportation Systems.',
    url: 'https://arxiv.org/abs/1901.06340',
  },
  {
    id: 'li2023hrsegnet',
    label: 'Li et al. 2023',
    citation:
      'Li Y. et al. (2023). Real-time high-resolution neural network with semantic guidance for crack segmentation. Automation in Construction 156, 105112.',
    url: 'https://arxiv.org/abs/2307.00270',
  },
  {
    id: 'liu2021crackformer',
    label: 'Liu et al. 2021',
    citation:
      'Liu H., Miao X., Mertz C., Xu C., Kong H. (2021). CrackFormer: Transformer Network for Fine-Grained Crack Detection. ICCV 2021, 3763-3772.',
    doi: '10.1109/ICCV48922.2021.00376',
  },
  {
    id: 'liu2023crackformer2',
    label: 'Liu et al. 2023',
    citation:
      'Liu H. et al. (2023). CrackFormer Network for Pavement Crack Segmentation. IEEE Transactions on Intelligent Transportation Systems 24(9), 9240-9252.',
    doi: '10.1109/TITS.2023.3266776',
  },
  {
    id: 'shit2021cldice',
    label: 'Shit et al. 2021',
    citation:
      'Shit S. et al. (2021). clDice: a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. CVPR 2021.',
    url: 'https://arxiv.org/abs/2003.07311',
  },
  {
    id: 'cha2018damage',
    label: 'Cha et al. 2018',
    citation:
      'Cha Y.-J., Choi W., Suh G., Mahmoudkhani S., Buyukozturk O. (2018). Autonomous Structural Visual Inspection Using Region-Based Deep Learning for Detecting Multiple Damage Types. Computer-Aided Civil and Infrastructure Engineering.',
    doi: '10.1111/mice.12334',
  },
  // --- foundation era ---
  {
    id: 'kirillov2023sam',
    label: 'Kirillov et al. 2023',
    citation: 'Kirillov A. et al. (2023). Segment Anything. ICCV 2023.',
    url: 'https://arxiv.org/abs/2304.02643',
  },
  {
    id: 'ravi2024sam2',
    label: 'Ravi et al. 2024',
    citation: 'Ravi N. et al. (2024). SAM 2: Segment Anything in Images and Videos.',
    url: 'https://arxiv.org/abs/2408.00714',
  },
  {
    id: 'oquab2023dinov2',
    label: 'Oquab et al. 2023',
    citation: 'Oquab M. et al. (2023). DINOv2: Learning Robust Visual Features without Supervision.',
    url: 'https://arxiv.org/abs/2304.07193',
  },
  {
    id: 'ge2024cracksam',
    label: 'CrackSAM 2024',
    citation:
      'CrackSAM: fine-tuning the Segment Anything Model for crack segmentation via parameter-efficient adaptation (adapter and LoRA on a frozen SAM encoder).',
    url: 'https://arxiv.org/abs/2312.04233',
  },
  {
    id: 'sac2025',
    label: 'Segment Any Crack 2025',
    citation:
      'Segment Any Crack: normalization-only tuning of SAM for crack segmentation; F1 61.22 on OmniCrack30k.',
    url: 'https://arxiv.org/abs/2504.14138',
  },
  {
    id: 'benz2024omnicrack',
    label: 'Benz & Rodehorst 2024',
    citation:
      'Benz C., Rodehorst V. (2024). OmniCrack30k: A Benchmark for Crack Segmentation and the Reasonable Effectiveness of Transfer Learning. CVPR Workshops 2024, 3876-3886.',
    url: 'https://github.com/ben-z-original/omnicrack30k',
  },
  // --- anomaly detection ---
  {
    id: 'bergmann2019mvtec',
    label: 'Bergmann et al. 2019/2021',
    citation:
      'Bergmann P., Fauser M., Sattlegger D., Steger C. (2019). MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. CVPR 2019; extended IJCV 129(4), 2021 (PRO metric).',
    doi: '10.1007/s11263-020-01400-4',
  },
  {
    id: 'roth2022patchcore',
    label: 'Roth et al. 2022',
    citation: 'Roth K. et al. (2022). Towards Total Recall in Industrial Anomaly Detection (PatchCore). CVPR 2022.',
    url: 'https://arxiv.org/abs/2106.08265',
  },
  {
    id: 'batzner2024efficientad',
    label: 'Batzner et al. 2024',
    citation: 'Batzner K., Heckler L., Koenig R. (2024). EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies. WACV 2024.',
    url: 'https://arxiv.org/abs/2303.14535',
  },
  {
    id: 'akcay2022anomalib',
    label: 'Akcay et al. 2022',
    citation: 'Akcay S. et al. (2022). Anomalib: A Deep Learning Library for Anomaly Detection. ICIP 2022.',
    url: 'https://arxiv.org/abs/2202.08341',
  },
  {
    id: 'zou2022visa',
    label: 'Zou et al. 2022',
    citation:
      'Zou Y., Jeong J., Pemula L., Zhang D., Dabeer O. (2022). SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation (VisA). ECCV 2022.',
    url: 'https://arxiv.org/abs/2207.14315',
  },
  {
    id: 'heckler2026mvtec2',
    label: 'Heckler-Kram et al. 2026',
    citation:
      'Heckler-Kram L., Neudeck J.-H., Scheler U., Koenig R., Steger C. (2026). The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection. IJCV.',
    doi: '10.1007/s11263-026-02743-0',
  },
  // --- datasets ---
  {
    id: 'dorafshan2018sdnet',
    label: 'Dorafshan et al. 2018',
    citation:
      'Dorafshan S., Thomas R.J., Maguire M. (2018). SDNET2018: An annotated image dataset for non-contact concrete crack detection using deep convolutional neural networks. Data in Brief 21.',
    doi: '10.1016/j.dib.2018.11.015',
  },
  {
    id: 'ozgenel2018',
    label: 'Ozgenel & Gonenc Sorguc 2018',
    citation:
      'Ozgenel C.F., Gonenc Sorguc A. (2018). Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings. ISARC 2018, Berlin. Dataset: Mendeley Data.',
    doi: '10.17632/5y9wdsg2zt.2',
  },
  {
    id: 'mundt2019codebrim',
    label: 'Mundt et al. 2019',
    citation:
      'Mundt M., Majumder S., Murali S., Panetsos P., Ramesh V. (2019). Meta-learning Convolutional Neural Architectures for Multi-target Concrete Defect Classification with the COncrete DEfect BRidge IMage Dataset (CODEBRIM). CVPR 2019.',
    url: 'https://arxiv.org/abs/1904.08486',
  },
  {
    id: 'flotzinger2024dacl10k',
    label: 'Flotzinger et al. 2024',
    citation:
      'Flotzinger J., Roesch P.J., Braml T. (2024). dacl10k: Benchmark for Semantic Bridge Damage Segmentation. WACV 2024.',
    url: 'https://arxiv.org/abs/2309.00460',
  },
  {
    id: 'kulkarni2022crackseg9k',
    label: 'Kulkarni et al. 2022',
    citation:
      'Kulkarni S., Singh S., Balakrishnan D., Sharma S., Devunuri S., Korlapati S.C.R. (2022). CrackSeg9k: A Collection and Benchmark for Crack Segmentation Datasets and Frameworks. ECCV 2022 Workshops.',
    doi: '10.1007/978-3-031-25082-8_12',
  },
  {
    id: 'ye2021bcl',
    label: 'Ye et al. 2021',
    citation:
      'Ye X.W., Jin T., Li Z.X., Ma S.Y., Ding Y., Ou Y.H. (2021). Structural Crack Detection from Benchmark Data Sets Using Pruned Fully Convolutional Networks (Bridge Crack Library). Journal of Structural Engineering 147(11), 04721008.',
    doi: '10.7910/DVN/RURXSH',
  },
  // --- quantification, monitoring, deformation ---
  {
    id: 'aci224r01',
    label: 'ACI 224R-01',
    citation:
      'ACI Committee 224 (2001, reapproved). Control of Cracking in Concrete Structures (ACI 224R-01). American Concrete Institute. Guide table of tolerable crack widths; the document itself cautions that width is not always a reliable corrosion indicator.',
  },
  {
    id: 'en1992',
    label: 'EN 1992-1-1:2004',
    citation:
      'CEN (2004). Eurocode 2: Design of concrete structures, Part 1-1, Table 7.1N: recommended limiting calculated crack widths (National Annex dependent).',
  },
  {
    id: 'paris1963',
    label: 'Paris & Erdogan 1963',
    citation:
      'Paris P.C., Erdogan F. (1963). A Critical Analysis of Crack Propagation Laws. Journal of Basic Engineering 85(4), 528-534.',
    doi: '10.1115/1.3656900',
  },
  {
    id: 'pan2009dic',
    label: 'Pan et al. 2009',
    citation:
      'Pan B., Qian K., Xie H., Asundi A. (2009). Two-dimensional digital image correlation for in-plane displacement and strain measurement: a review. Measurement Science and Technology 20, 062001.',
    doi: '10.1088/0957-0233/20/6/062001',
  },
  {
    id: 'olufsen2020mudic',
    label: 'Olufsen et al. 2020',
    citation:
      'Olufsen S.N., Andersen M.E., Fagerholt E. (2020). muDIC: An open-source toolkit for digital image correlation. SoftwareX 11, 100391.',
    doi: '10.1016/j.softx.2019.100391',
  },
  {
    id: 'jiang2023opencorr',
    label: 'Jiang 2023',
    citation:
      'Jiang Z. (2023). OpenCorr: An open source library for research and development of digital image correlation. Optics and Lasers in Engineering 165, 107566.',
    doi: '10.1016/j.optlaseng.2023.107566',
  },
  {
    id: 'zhu2023crackpropnet',
    label: 'Zhu & Al-Qadi 2023',
    citation:
      'Zhu Z., Al-Qadi I.L. (2023). Automated crack propagation measurement on asphalt concrete specimens using an optical flow-based deep neural network (CrackPropNet). International Journal of Pavement Engineering 24(1).',
    doi: '10.1080/10298436.2023.2186407',
  },
  {
    id: 'melching2022',
    label: 'Melching et al. 2022',
    citation:
      'Melching D., Strohmann T., Requena G., Breitbarth E. (2022). Explainable machine learning for precise fatigue crack tip detection. Scientific Reports 12, 9513.',
    doi: '10.1038/s41598-022-13275-1',
  },
  // --- surveys ---
  {
    id: 'spencer2019',
    label: 'Spencer et al. 2019',
    citation:
      'Spencer B.F., Hoskere V., Narazaki Y. (2019). Advances in Computer Vision-Based Civil Infrastructure Inspection and Monitoring. Engineering 5(2), 199-222.',
    doi: '10.1016/j.eng.2018.11.030',
  },
  {
    id: 'zhang2025review',
    label: 'Zhang et al. 2025',
    citation:
      'Zhang X., Wang H., Hsieh Y.-A., Yang Z., Yezzi A., Tsai Y.-C. (2025). Deep Learning for Crack Detection: A Review of Learning Paradigms, Generalizability, and Datasets.',
    url: 'https://arxiv.org/abs/2508.10256',
  },
];
