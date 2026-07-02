# Behavior-Aware Feature Engineering for Credit Card Fraud Detection

*A Deep Learning Framework Built on Homogeneity-Oriented Behavior Analysis (HOBA)*

**ECO723 Group Project | Replication & Extension on a 1.85M-Transaction Dataset**

---

## Executive Summary

Credit card issuers face a dual cost problem: undetected fraud drains money directly, while overly aggressive fraud models that generate false positives quietly drain money too, through declined legitimate transactions, support load, and customer attrition. This project tests whether richer, behavior-aware features (HOBA) combined with deep learning can catch materially more fraud than a conventional Recency-Frequency-Monetary (RFM) feature set, without increasing the false-positive burden on genuine customers.

Using a 1.85-million-transaction dataset, we engineered a 77-feature HOBA representation and a 4-feature RFM baseline, then trained six classifiers (Random Forest, SVM, a shallow neural network, a 1D CNN, an Elman RNN, and a two-layer Deep Belief Network) on both. HOBA features improved AUC for every model, by as much as +0.13 for the weaker learners. At a strict 1% false-positive budget, the best HOBA-based model (Random Forest) caught 82.6% of fraud while keeping accuracy above 98.9%. Full results, including performance at 1% and 3% false-positive tolerances (mirroring the evaluation protocol of Zhang, Han, Xu & Wang, Information Sciences, 2021), are reported below.

---

## 1. The Business Problem

Credit card issuers lose billions of dollars a year to fraud — the Nilson Report placed credit and debit card fraud losses at $16.31 billion in 2014, up 19% year-over-year. But the deeper operational problem is not only missed fraud: it is false alarms. Legacy rule-based and shallow-feature systems, built on a handful of raw signals such as transaction amount, ZIP code, and distance from home, cannot reliably tell an unusual-but-legitimate purchase from a genuinely fraudulent one.

Every false positive is a declined legitimate transaction. It damages customer trust, generates support-desk load, and increases the risk of card abandonment. A fraud model that is very good at catching fraud but careless with false positives can end up costing the issuer more in customer attrition than the fraud it prevents. Conversely, a model tuned only to minimize false positives will let fraud through. Both failure modes are expensive, which is why fraud detection is evaluated at a fixed, business-realistic false-positive tolerance (typically 1–3%) rather than by accuracy alone — with a dataset that is over 99% legitimate transactions, a model that flags nothing at all is already ~99% "accurate" and completely useless.

The objective of this project was therefore twofold:

- Build a richer, behavior-aware feature set that gives models enough context to separate genuine anomalies from normal-but-varied spending.
- Test whether deep learning architectures can convert that richer feature set into materially better fraud detection at a strict, business-realistic false-positive tolerance (1–3%).

---

## 2. Our Approach

We implemented and empirically compared two feature engineering philosophies on the same transaction-level dataset — a large-scale card-transaction dataset with approximately 1.85 million chronologically sorted transactions, split into a training set (1,296,675 transactions) and a test set (555,719 transactions):

- **RFM baseline (4 features)** — transaction amount, ZIP code, city population, and distance to merchant. This mirrors the traditional Recency/Frequency/Monetary approach used by most legacy rule engines and by the benchmark models in the HOBA reference paper.
- **HOBA — Homogeneity-Oriented Behavior Analysis (77 engineered features)** — rolling statistics computed per cardholder over 1-, 3-, 7- and 15-day windows, split out by transaction category, plus velocity counters and geographic-jump detectors, following the feature engineering methodology proposed by Zhang, Han, Xu & Wang (Information Sciences, 2021).

On top of both feature sets, the same six classifiers were trained and evaluated under identical conditions:

- **Random Forest (RF)** and **Support Vector Machine (SVM)** — the two most widely used traditional data-mining techniques in fraud detection.
- **Back-Propagation Neural Network (BPNN)** — a shallow, single-hidden-layer network representing the traditional artificial-neural-network benchmark.
- **1D Convolutional Neural Network (CNN)** — treats each cardholder's feature vector as a 1D signal and learns local patterns via convolution and pooling.
- **Elman Recurrent Neural Network (RNN)** — a recurrent architecture with a context layer that feeds hidden-layer activations back as additional input, giving the network a simple form of memory.
- **Two-layer Deep Belief Network (DBN)** — two stacked Restricted Boltzmann Machines (RBMs) pretrained layer-by-layer in an unsupervised fashion, then fine-tuned end-to-end with a class-weighted loss (BCEWithLogitsLoss) to correct for the heavy fraud/non-fraud imbalance (fraud is under 1.5% of transactions).

### 2.1 Core Pillars of the HOBA Feature Set

- **Spending habits** — rolling mean / sum / max of transaction amount per cardholder (e.g. "Is this amount unusual for this user this week?" rather than "Is this amount unusual in general?").
- **Pattern velocity** — count of transactions within a rolling 1-hour window, used to catch rapid-fire card testing, plus tighter time-window aggregations by merchant category.
- **Spatial logic** — haversine distance between cardholder and merchant location for the current transaction, plus the maximum distance travelled in a rolling 3-day window, to catch geographically impossible transaction sequences.
- **Category-conditioned aggregation** — rolling statistics are computed separately per merchant category rather than pooled together, so a cardholder's grocery spending pattern is not diluted by their travel spending pattern. This directly operationalizes the "homogeneity-oriented" principle of the reference paper: transactions of intrinsically different characteristics are analyzed in separate subspaces rather than lumped into one aggregate.
- **Rule-based flags** — binary indicators for abnormal transaction time (1am–5am), high-value transactions (>$500), and high-velocity behavior (more than 3 transactions in the past hour), reflecting the domain-knowledge-driven "rule-based strategy" the reference paper describes alongside transaction aggregation.

---

## 3. Data and Evaluation Framework

### 3.1 Essential Transaction Attributes

The raw dataset carries the transaction- and account-level fields summarized below, consistent with the attribute frame described in the HOBA reference paper (Table 1).

| Attribute | Description | Information Level |
|---|---|---|
| Card number (cc_num) | Unique credit card identifier used to group a cardholder's transaction history | Transaction Detail |
| Transaction amount (amt) | Dollar value of the transaction | Transaction Detail |
| Transaction date/time | Timestamp of the transaction, used for recency and rolling-window features | Transaction Detail |
| Merchant category (category) | The type of merchant / spending category (e.g. grocery, travel, gas) | Transaction Detail |
| Merchant location (merch_lat, merch_long) | Latitude/longitude of the merchant | Transaction Detail |
| Cardholder location (lat, long) | Latitude/longitude associated with the cardholder | Account Status |
| ZIP code / city population | Geographic and demographic context used by the RFM baseline | Account Status |
| is_fraud | Ground-truth fraud label used for supervised training and evaluation | Label |

> **Table 1 (paper-equivalent).** Field descriptions reconstructed from the feature engineering pipeline in the project notebook; the underlying dataset does not carry an explicit open-to-buy/credit-limit field as the original bank dataset in the reference paper did, so those two rows from the original paper's Table 1 do not apply here.

### 3.2 Evaluation Criteria

As in the reference paper, we do not rely on raw accuracy, since the dataset is over 99% legitimate transactions and a model that flags nothing would already score ~99%+ accuracy while catching no fraud. Instead we report Precision, Recall, F1-Measure, Accuracy and AUC, defined against the standard confusion matrix below, and — critically — we evaluate Precision, Recall, F1 and Accuracy at fixed, business-realistic false-positive-rate (FPR) budgets of 1% and 3%, rather than at the default 0.5 probability threshold.

| | Predicted: Positive (Fraud) | Predicted: Negative (Legitimate) |
|---|---|---|
| **Actual: Positive (Fraud)** | True Positive (TP) | False Negative (FN) |
| **Actual: Negative (Legitimate)** | False Positive (FP) | True Negative (TN) |

> **Table 2 (paper-equivalent).** Confusion-matrix definitions. Recall = TP/(TP+FN); Precision = TP/(TP+FP); F1 = 2·Precision·Recall/(Precision+Recall); FPR = FP/(FP+TN).

---

## 4. Results

### 4.1 Overall Discriminative Power (AUC)

The table below reports the Area-Under-the-ROC-Curve (AUC) achieved by each classifier on the held-out test set, comparing the 4-feature RFM baseline against the 77-feature HOBA set. This is the direct analogue of the AUC column in the reference paper's Table 3.

| Classifier | RFM AUC | HOBA AUC | Improvement |
|---|---|---|---|
| Random Forest | 0.9578 | 0.9805 | +0.0227 |
| BPNN (shallow NN) | 0.8286 | 0.9590 | +0.1304 |
| Deep Belief Network | 0.8320 | 0.9571 | +0.1251 |
| Elman RNN | 0.8247 | 0.9544 | +0.1297 |
| 1D CNN | 0.8942 | 0.9527 | +0.0585 |
| SVM | 0.8230 | 0.9522 | +0.1291 |

Two findings stand out. First, HOBA features lift every model, and the lift is largest for the models that started weakest on the sparse RFM set (BPNN, DBN, RNN and SVM all gain roughly 0.13 AUC) — richer behavioral context matters more when the base learner has less capacity to compensate for weak inputs. Second, Random Forest is the strongest model overall on both feature sets, and by a clear margin on RFM (0.9578 vs. the next-best CNN at 0.8942). This differs from the reference paper, where the DBN was the top performer on the bank's real transaction data; on this dataset, Random Forest consistently leads.

### 4.2 ROC Curves

![ROC curves by feature set and model family](Final_ROC_Curves_Complete.png)

*Figure 1. ROC curves by feature set and model family. Top-left: all classifiers on HOBA. Top-right: all classifiers on RFM. Bottom-left: deep learning models, HOBA (solid) vs. RFM (dashed). Bottom-right: traditional ML models, HOBA (solid) vs. RFM (dashed). Source: project notebook output (Final_ROC_Curves_Complete.png).*

The HOBA panels (left column) show curves hugging the top-left corner far more tightly than the RFM panels (right column), directly visualizing the AUC gains in Section 4.1. The bottom-left panel is especially informative: for CNN and RNN, the dashed (RFM) curves show a visible "step" — a flat stretch followed by a jump — around FPR 0.15–0.65, which corresponds to the recurring low-recall plateau seen in the RFM tables below.

### 4.3 Performance at a Strict 1% False-Positive Budget

AUC tells us about ranking quality across all thresholds, but the business only cares about performance at the false-positive rate it can tolerate. The tables below fix FPR at 1% — i.e., at most 1 in 100 legitimate cardholders is inconvenienced — and report how much fraud each model actually catches. This mirrors Table 5a in the reference paper.

**HOBA Features (1% FPR)**

| Classifier | F1-Measure | Precision | Recall | Accuracy |
|---|---|---|---|---|
| Random Forest | 0.375 | 24.28% | 82.56% | 98.94% |
| BPNN | 0.360 | 23.33% | 78.37% | 98.92% |
| Elman RNN | 0.346 | 22.50% | 74.87% | 98.91% |
| Deep Belief Network | 0.346 | 22.49% | 74.83% | 98.91% |
| 1D CNN | 0.341 | 22.23% | 73.66% | 98.90% |
| SVM | 0.338 | 22.01% | 72.45% | 98.90% |

**RFM Features (1% FPR)**

| Classifier | F1-Measure | Precision | Recall | Accuracy |
|---|---|---|---|---|
| Random Forest | 0.300 | 19.65% | 63.08% | 98.86% |
| BPNN | 0.252 | 16.97% | 48.86% | 98.88% |
| Deep Belief Network | 0.242 | 16.04% | 48.95% | 98.81% |
| Elman RNN | 0.241 | 16.00% | 48.62% | 98.82% |
| 1D CNN | 0.240 | 15.94% | 48.76% | 98.81% |
| SVM | 0.236 | 15.68% | 47.93% | 98.80% |

> **Table 5a (paper-equivalent).** At this strict operating point, Random Forest leads on every metric under both feature sets, catching over 82% of fraud on HOBA while keeping accuracy above 98.9%. On the sparse RFM set, no model exceeds 64% recall — the ceiling effect visible as the early plateau in the top-right ROC panel.

### 4.4 Performance at a Relaxed 3% False-Positive Budget

Relaxing the false-positive tolerance to 3% — the maximum most fraud teams consider acceptable — lets every model catch substantially more fraud, at the cost of lower precision. This mirrors Table 5b in the reference paper.

**HOBA Features (3% FPR)**

| Classifier | F1-Measure | Precision | Recall | Accuracy |
|---|---|---|---|---|
| Random Forest | 0.188 | 10.51% | 90.68% | 96.99% |
| BPNN | 0.181 | 10.09% | 86.34% | 96.98% |
| Deep Belief Network | 0.178 | 9.96% | 85.17% | 96.97% |
| Elman RNN | 0.178 | 9.94% | 84.85% | 96.98% |
| 1D CNN | 0.176 | 9.84% | 84.15% | 96.96% |
| SVM | 0.176 | 9.81% | 83.36% | 96.98% |

**RFM Features (3% FPR)**

| Classifier | F1-Measure | Precision | Recall | Accuracy |
|---|---|---|---|---|
| Deep Belief Network | 0.159 | 8.91% | 74.31% | 96.97% |
| BPNN | 0.159 | 8.87% | 74.41% | 96.95% |
| Elman RNN | 0.158 | 8.87% | 74.31% | 96.95% |
| 1D CNN | 0.158 | 8.82% | 74.08% | 96.94% |
| Random Forest | 0.157 | 8.78% | 74.55% | 96.91% |
| SVM | 0.152 | 8.53% | 72.07% | 96.91% |

> **Table 5b (paper-equivalent).** At 3% FPR, Random Forest still leads recall on HOBA (90.68%), but on RFM the models converge to a tight band (~72–75% recall) — the flat plateau visible in the top-right ROC panel where FPR runs from roughly 0.15 to 0.65 with almost no TPR gain.

### 4.5 Statistical Significance of Improvements

The reference paper additionally reports McNemar's test results comparing each HOBA/deep-learning classifier against each RFM baseline classifier (its Table 4), with all comparisons significant at p < 0.01. The project notebook provided to us does not include a McNemar's test or the raw per-transaction prediction arrays needed to compute one, so we cannot reproduce that table from the materials available. Given the sample size here (555,719 test transactions) and the consistent, non-trivial AUC gaps in Section 4.1 (+0.02 to +0.13), we would expect the same qualitative conclusion — that HOBA-based classifiers significantly outperform RFM-based ones — but this has not been formally tested on this dataset and should not be asserted as fact without the underlying computation.

---

## 5. Business Impact

- Moving from the 4-feature RFM baseline to the 77-feature HOBA representation raised AUC by up to 0.13 for the weaker learners and by 0.023 even for the already-strong Random Forest — confirming that richer behavioral context, not just a bigger model, is the main lever for fraud detection quality.
- At a 1% false-positive budget (a tolerance the business can realistically accept), the best HOBA-based model catches over 82% of fraudulent transactions while disturbing fewer than roughly 1% of legitimate cardholders.
- Because Random Forest matched or beat every deep learning architecture tested, it is the recommended first production candidate: comparable or better fraud capture, materially lower training/inference cost, and easier explainability for regulatory and dispute-resolution purposes. Deep architectures — particularly the DBN, which needs unsupervised pretraining — can be kept as a research track for future ensembling rather than the initial production model.
- The RFM baseline hits a hard recall ceiling around 63–75% even at a relaxed 3% FPR tolerance, regardless of model choice — this is a feature-set limitation, not a model-capacity limitation, and is exactly the gap HOBA is designed to close.

---

## 6. Limitations & Future Work

- Model training used a downsampled SVM training set and a small number of epochs for the neural architectures (CNN, RNN, DBN) to keep runtime tractable; results for these models may improve with longer training and hyperparameter tuning.
- Two different HOBA feature counts appear across the notebook's experiments — an earlier 9-feature "frozen" HOBA set and the final 77-feature set used for the results in this report. All figures reported here use the final 77-feature version and its matching AUC/FPR tables; readers should not mix numbers from the two runs.
- An ensemble of Random Forest and the deep architectures (e.g., stacking or a weighted blend) may outperform any single model and is a natural next step.
- Computational cost of generating 77 rolling-window features in real time was not benchmarked here and should be assessed before production deployment, consistent with a limitation flagged in the HOBA reference paper itself.
- This dataset differs from the one used in the original HOBA paper (a real bank dataset of ~115K transactions with 1,410 candidate HOBA variables); our AUC and recall figures are not directly comparable to the paper's Table 3/5a/5b values and should be read as an independent replication on a different dataset, not a reproduction of the paper's exact numbers.

---

## 7. Data Provenance

All AUC, precision, recall, F1 and accuracy figures, the model architectures, and the ROC curve figure in this report were taken directly from the executed project notebook (Creditcardfraud_samefeatures) and its printed output tables. The qualitative framing of the business problem, the four HOBA pillars, and the reference-paper citation were adapted from the prior project report and cross-checked against both the notebook and the original HOBA paper (Zhang, Han, Xu & Wang, Information Sciences 557 (2021) 302–316) for consistency.

## 8. Author Contributions

- **Rituraj Singh (240877):** Lead Engineer & Technical Director. Engineered the custom PyTorch deep learning architectures (1D CNN, Elman RNN, and Deep Belief Network) from scratch. Implemented the Contrastive Divergence pre-training loop, executed the model training pipelines, and defined the technical structure and content requirements and authored the final report.
- **Nipun Basyal (250726):** Lead Technical Writer. Authored the final written report based on the engineering outlines. Standardized the feature inputs across the datasets to ensure strict experimental control, and structured the final performance tables and business impact analysis for submission.