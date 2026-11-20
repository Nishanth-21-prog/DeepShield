
# DeepShield - Dataset Research

## Dataset
- Name: CIC-IDS2017
- Source: Canadian Institute for Cybersecurity
- Recording period: July 3-7, 2017
- Data type: Network flow records
- Feature extraction: CICFlowMeter

## Classification Tasks
1. Binary classification: Benign vs Attack
2. Multiclass classification: Benign vs individual attack types

## Attack Families
- Brute Force
- DoS / DDoS
- Web Attacks
- Infiltration
- Botnet
- PortScan
- Heartbleed

## Important Feature Groups
- Flow duration
- Packet counts
- Byte counts
- Packet lengths
- Inter-arrival times
- TCP flags

## Research Challenges
- Class imbalance
- Duplicate records
- Identifier leakage
- Train/test contamination
- Generalization to unseen traffic

## Initial Dataset Choice
MachineLearningCSV.zip

## Next Steps
- Download original CSV files
- Inspect CSV structure
- Identify missing and infinite values
- Inspect class distribution
- Design leakage-aware train/validation/test splits


## Dataset Exploration - Day 1 Results

### Dataset Statistics
- Total CSV files: 8
- Total records: 2,830,743
- Columns per CSV: 79
- Input feature columns: 78
- Target column: Label
- Unique labels: 15

### Class Distribution

| Label | Records |
|---|---:|
| BENIGN | 2,273,097 |
| DoS Hulk | 231,073 |
| PortScan | 158,930 |
| DDoS | 128,027 |
| DoS GoldenEye | 10,293 |
| FTP-Patator | 7,938 |
| SSH-Patator | 5,897 |
| DoS slowloris | 5,796 |
| DoS Slowhttptest | 5,499 |
| Bot | 1,966 |
| Web Attack - Brute Force | 1,507 |
| Web Attack - XSS | 652 |
| Infiltration | 36 |
| Web Attack - SQL Injection | 21 |
| Heartbleed | 11 |

### Initial Observations
1. Dataset contains significant class imbalance.
2. BENIGN traffic represents approximately 80.3% of all records.
3. Some attack classes contain very few samples.
4. Potential label encoding issue observed in Web Attack labels.
5. Further investigation of missing values, infinite values and duplicates is required.

### Initial Model Strategy
- First experiment: Binary classification.
- Second experiment: Multiclass classification.
- Evaluation: Precision, Recall, F1-score, confusion matrix and false-positive rate.

### Pending Research
- Missing value analysis
- Infinite value analysis
- Duplicate record analysis
- Feature correlation
- Data leakage investigation
