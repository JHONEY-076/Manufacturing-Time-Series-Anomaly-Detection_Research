# 4.1. Dataset construction and preprocessing

The experiments used the KAMP Battery Pack Welding Process Dataset, which contains multivariate time-series records from a welding process. Figure X illustrates the conversion of the original manufacturing records into labeled windows for defect detection. The preprocessing procedure comprises chronological ordering, input feature selection and standardization, sliding-window segmentation, and window-level labeling.

The original process variables include Speed, Length, RealPower, SetFrequency, SetDuty, SetPower, and GateOnTime, together with the timestamp WorkingTime. Records were arranged chronologically within each recording using WorkingTime. Five variables—RealPower, GateOnTime, SetPower, Speed, and Length—were retained as model inputs. SetFrequency and SetDuty were excluded because they remained constant in the analyzed recordings. WorkingTime was used to establish temporal order and was excluded from the predictor set, together with record identifiers and labels.

The selected variables were standardized using feature-wise means and standard deviations estimated from the original normal training records. For an observation $x_{t,f}$ at time position $t$ and feature $f$, the standardized value was computed as

$$
z_{t,f}=\frac{x_{t,f}-\mu_f^{\mathrm{tr}}}{\sigma_f^{\mathrm{tr}}},
$$

where $\mu_f^{\mathrm{tr}}$ and $\sigma_f^{\mathrm{tr}}$ denote the mean and standard deviation of feature $f$ in the normal training records. These parameters were fixed and applied to the defect seed, validation, and final test data.

Each standardized recording was segmented into overlapping windows of 30 consecutive observations with a stride of one observation. The first three windows therefore covered positions 1–30, 2–31, and 3–32, respectively. Windows were constructed separately within each source recording. With five input variables, each window was represented as $\mathbf X_i\in\mathbb R^{30\times5}$. The temporal structure was retained for data generation, while each window was subsequently flattened into a 150-dimensional vector for RandomForest classification.

Window labels were assigned from the binary labels of the constituent observations. Let $b_t\in\{0,1\}$ denote the label of observation $t$, where 0 indicates a normal observation and 1 indicates a defect observation. For a window beginning at position $i$, its label was defined as

$$
y_i=\max_{\ell=0,\ldots,29}b_{i+\ell}.
$$

Thus, a window was labeled as defective if it contained at least one observation labeled as defective. A window was labeled as normal when all constituent observations were normal. Records from normal-only files were assigned observation labels of 0, and the available observation labels were retained for the labeled defect recordings.

The dataset partitions used in this study were constructed for normal training, defect seed training, validation, and final evaluation. The original normal training file provided the normal training pool. Recordings from the original test files were reorganized to construct the defect seed, validation, and final test sets used in this study. Table X summarizes the resulting window counts.

**Table X. Composition of the windowed dataset.**

| Dataset partition | Normal windows | Defect windows | Total windows |
| --- | ---: | ---: | ---: |
| Normal training pool | 135,730 | 0 | 135,730 |
| Defect training seeds | 0 | 247 | 247 |
| Validation set | 1,283 | 164 | 1,447 |
| Final test set | 2,785 | 296 | 3,081 |

For the principal augmentation comparisons, 5,000 normal windows were sampled from the normal training pool and combined with all 247 real defect seed windows, yielding an initial normal-to-defect ratio of approximately 20.24:1. Synthetic defect samples were added to the training data according to each augmentation condition. The validation and final test sets contained real windows. Decision thresholds were selected using the validation set, and the resulting classifiers and thresholds were applied to the final test set. All sample counts are reported at the window level.

**Fig. X. Dataset construction and preprocessing for welding time-series defect detection.** Original manufacturing records are ordered by WorkingTime and segmented into windows of 30 consecutive observations with a stride of one. Each window is labeled as defective if it contains at least one defect-labeled observation; otherwise, it is labeled as normal.

---

## Author notes — resolve before submission

- In the figure, replace the second occurrence of “Time-Order Sorting” with “Sliding-window segmentation”. Add feature selection and standardization to align the diagram with the procedure described in the text. The model input contains five process variables.
- The current stored partitions share 136 source observations between the defect seed and validation sets and one source observation between the validation and final test sets. These counts refer to unique source-file row positions. Resolve the overlap and repeat the affected evaluation before claiming that the partitions are independent; otherwise, disclose this limitation in the manuscript.
- Data counts were checked against `data/augmentation_split/split_summary.csv`; overlap was checked using the corresponding metadata files. Standardization was verified from `tools/research11_export_api_models.py`, and window construction and label assignment were checked in `data/raw_data/Research.ipynb`.
