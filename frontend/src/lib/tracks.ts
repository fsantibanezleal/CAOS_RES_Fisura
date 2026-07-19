// The lab's spine: 7 method tracks and 16 planned cases, as validated in plans/fisura/plan.md
// (2026-07-18). Status flips per unit as engines land; nothing here claims more than exists.

export type TrackId =
  | 'classical'
  | 'learned'
  | 'foundation'
  | 'anomaly'
  | 'multiclass'
  | 'quantify'
  | 'monitor';

export interface Track {
  id: TrackId;
  index: number;
  tone: string; // CSS var reference
  en: string;
  es: string;
  blurb_en: string;
  blurb_es: string;
}

export const TRACKS: Track[] = [
  {
    id: 'classical', index: 1, tone: 'var(--fs-classical)',
    en: 'Classical pipelines', es: 'Pipelines clásicos',
    blurb_en: 'Illumination flattening, ridge filters, morphology, minimal paths: deterministic and inspectable, the honest floor.',
    blurb_es: 'Aplanado de iluminación, filtros de crestas, morfología, caminos mínimos: determinista e inspeccionable, el piso honesto.',
  },
  {
    id: 'learned', index: 2, tone: 'var(--fs-learned)',
    en: 'Learned segmentation', es: 'Segmentación aprendida',
    blurb_en: 'U-Net to transformer crack networks, trained in-repo on open datasets with published recipes.',
    blurb_es: 'De U-Net a redes transformer de grietas, entrenadas en el repo sobre datasets abiertos con recetas publicadas.',
  },
  {
    id: 'foundation', index: 3, tone: 'var(--fs-foundation)',
    en: 'Foundation models', es: 'Modelos fundacionales',
    blurb_en: 'SAM adapters, frozen DINOv2 features, zero-shot rows: the frontier, evaluated without the hype.',
    blurb_es: 'Adaptadores de SAM, features congeladas de DINOv2, filas zero-shot: la frontera, evaluada sin humo.',
  },
  {
    id: 'anomaly', index: 4, tone: 'var(--fs-anomaly)',
    en: 'Anomaly detection', es: 'Detección de anomalías',
    blurb_en: 'Train on good surfaces only; flag what deviates. Includes the concrete-transfer study the literature lacks.',
    blurb_es: 'Entrenar solo con superficies sanas; marcar lo que se desvía. Incluye el estudio de transferencia a hormigón que falta en la literatura.',
  },
  {
    id: 'multiclass', index: 5, tone: 'var(--fs-multiclass)',
    en: 'Multi-class damage', es: 'Daño multiclase',
    blurb_en: 'Beyond binary cracks: spalling, efflorescence, exposed bars, corrosion, on bridge inspection benchmarks.',
    blurb_es: 'Más allá de la grieta binaria: descascaramiento, eflorescencia, barras expuestas, corrosión, en benchmarks de inspección de puentes.',
  },
  {
    id: 'quantify', index: 6, tone: 'var(--fs-quantify)',
    en: 'Quantification', es: 'Cuantificación',
    blurb_en: 'The flagship: calibrated width, length, orientation, density and severity context. Masks become engineering numbers.',
    blurb_es: 'El buque insignia: ancho calibrado, largo, orientación, densidad y contexto de severidad. Las máscaras se vuelven números de ingeniería.',
  },
  {
    id: 'monitor', index: 7, tone: 'var(--fs-monitor)',
    en: 'Monitoring + deformation', es: 'Monitoreo + deformación',
    blurb_en: 'Change between inspection epochs, bench growth curves, and 2D digital image correlation.',
    blurb_es: 'Cambio entre épocas de inspección, curvas de crecimiento en banco, y correlación digital de imágenes 2D.',
  },
];

export type CaseStatus = 'todo' | 'building' | 'live' | 'replay';

export interface PlannedCase {
  id: string;
  track: TrackId;
  status: CaseStatus;
  en: string;
  es: string;
  data_en: string;
  data_es: string;
}

export const PLANNED_CASES: PlannedCase[] = [
  { id: 'bcl_examples', track: 'classical', status: 'replay', en: 'BCL examples workbench', es: 'Banco de ejemplos BCL', data_en: 'Committed CC0/CC BY patches through ladder L0-L5; live in the App', data_es: 'Parches CC0/CC BY versionados por la escalera L0-L5; vivo en la App' },
  { id: 'synthetic_battery', track: 'quantify', status: 'replay', en: 'Synthetic validation battery', es: 'Batería sintética de validación', data_en: 'Exact ground truth: ladder regression gate + width-estimator accuracy', data_es: 'Ground truth exacto: compuerta de regresión de la escalera + exactitud de estimadores de ancho' },
  { id: 'asphalt-cfd', track: 'classical', status: 'todo', en: 'Asphalt: CFD anchors', es: 'Asfalto: anclas CFD', data_en: 'CrackForest CFD; the published classical ladder floor-to-ceiling', data_es: 'CrackForest CFD; la escalera clásica publicada de piso a techo' },
  { id: 'pavement-crack500', track: 'classical', status: 'todo', en: 'Pavement: Crack500', es: 'Pavimento: Crack500', data_en: 'Crack500; classical vs learned, the protocol-duality exhibit', data_es: 'Crack500; clásico vs aprendido, la exhibición de dualidad de protocolo' },
  { id: 'concrete-deepcrack537', track: 'classical', status: 'todo', en: 'Concrete: DeepCrack-537', es: 'Hormigón: DeepCrack-537', data_en: 'Mixed surfaces; thin-structure loss ablations', data_es: 'Superficies mixtas; ablaciones de pérdidas para estructuras finas' },
  { id: 'thin-lowcontrast', track: 'classical', status: 'todo', en: 'Thin cracks at the limit', es: 'Grietas finas al límite', data_en: 'Laser-illuminated and stone sets; ridge filters at their limit', data_es: 'Sets con iluminación láser y piedra; filtros de crestas al límite' },
  { id: 'sdnet-classify', track: 'learned', status: 'todo', en: 'SDNET2018 classification', es: 'Clasificación SDNET2018', data_en: '56k patches; leakage-safe splits, imbalance, label-noise honesty', data_es: '56k parches; splits sin fuga, desbalance, honestidad sobre ruido de etiquetas' },
  { id: 'ozgenel-saturation', track: 'learned', status: 'todo', en: 'The saturated benchmark', es: 'El benchmark saturado', data_en: 'Ozgenel 40k; why 99.9 percent accuracy teaches little', data_es: 'Ozgenel 40k; por qué 99.9 por ciento de exactitud enseña poco' },
  { id: 'crackseg9k-ladder', track: 'learned', status: 'building', en: 'CrackSeg9k learned ladder', es: 'Escalera aprendida CrackSeg9k', data_en: 'U-Net, DeepLabV3+, SegFormer-B2 trained (ladder A); HrSegNet + CrackFormer-II next', data_es: 'U-Net, DeepLabV3+, SegFormer-B2 entrenados (escalera A); HrSegNet + CrackFormer-II siguen' },
  { id: 'learned_on_examples', track: 'learned', status: 'replay', en: 'Learned ladder A on the examples', es: 'Escalera aprendida A en los ejemplos', data_en: 'Three trained architectures on the same committed patches as the classical ladder', data_es: 'Tres arquitecturas entrenadas sobre los mismos parches versionados que la escalera clásica' },
  { id: 'foundation-adapters', track: 'foundation', status: 'todo', en: 'Foundation adapters', es: 'Adaptadores fundacionales', data_en: 'SAM LoRA, norm-only tuning, SAM 2.1, DINOv2-linear, zero-shot', data_es: 'SAM LoRA, ajuste solo de normalización, SAM 2.1, DINOv2-lineal, zero-shot' },
  { id: 'dacl10k-damage', track: 'multiclass', status: 'todo', en: 'dacl10k bridge damage', es: 'Daño de puentes dacl10k', data_en: '19-class multi-label segmentation of real inspection imagery', data_es: 'Segmentación multietiqueta de 19 clases en imágenes reales de inspección' },
  { id: 'codebrim-detect', track: 'multiclass', status: 'todo', en: 'CODEBRIM detection', es: 'Detección CODEBRIM', data_en: 'Damage bounding boxes with permissive detectors', data_es: 'Cajas de daño con detectores de licencia permisiva' },
  { id: 'anomaly-industrial', track: 'anomaly', status: 'todo', en: 'Industrial anomaly ladder', es: 'Escalera industrial de anomalías', data_en: 'VisA + KolektorSDD2; PatchCore to EfficientAD, AU-PRO honesty', data_es: 'VisA + KolektorSDD2; de PatchCore a EfficientAD, honestidad AU-PRO' },
  { id: 'anomaly-concrete', track: 'anomaly', status: 'todo', en: 'Anomaly on concrete', es: 'Anomalías en hormigón', data_en: 'Good-only training on uncracked patches; the missing head-to-head', data_es: 'Entrenamiento solo-sano en parches sin grieta; el head-to-head que falta' },
  { id: 'width_bench', track: 'quantify', status: 'replay', en: 'Width measurement bench', es: 'Banco de medición de ancho', data_en: 'Three estimators vs exact truth; intensity FWHM within 0.007 px; mm calibrated', data_es: 'Tres estimadores vs verdad exacta; FWHM por intensidad a 0.007 px; calibrado a mm' },
  { id: 'severity_grading', track: 'quantify', status: 'replay', en: 'Severity context', es: 'Contexto de severidad', data_en: 'mm percentiles vs ACI 224R-01 and EC2 7.1N bands, caveats carried in-app', data_es: 'Percentiles en mm vs bandas ACI 224R-01 y EC2 7.1N, con sus caveats en la app' },
  { id: 'growth-monitoring', track: 'monitor', status: 'todo', en: 'Growth between epochs', es: 'Crecimiento entre épocas', data_en: 'Registered pairs; per-branch width deltas, bench a(N) curves', data_es: 'Pares registrados; deltas de ancho por rama, curvas a(N) de banco' },
  { id: 'dic-deformation', track: 'monitor', status: 'todo', en: 'DIC deformation', es: 'Deformación DIC', data_en: 'muDIC virtual experiments; strain fields and crack opening', data_es: 'Experimentos virtuales muDIC; campos de deformación y apertura de grieta' },
];

export const trackById = (id: TrackId) => TRACKS.find((t) => t.id === id)!;
