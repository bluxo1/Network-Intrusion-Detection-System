# KDDTest+ R2L error analysis

**Evaluation only.** KDDTest+ is a held-out test set. Do not use this report to select a threshold, feature set, checkpoint, or hyperparameters; make those choices using training/validation data.

Binary threshold: 0.248803 (saved/active inference threshold).

| Stage | Count |
| --- | ---: |
| True R2L records | 2885 |
| Rejected by binary gate as Normal | 2431 |
| Passed binary gate | 454 |
| Passed gate, assigned wrong attack family | 40 |
| Correctly assigned R2L | 414 |

R2L gate recall: 15.7%; conditional family accuracy: 91.2%; end-to-end R2L recall: 14.4%.
Normal false alarms at the same threshold: 299/9711 (3.1%).

## Attack subtypes

'Seen' means the raw subtype occurs anywhere in KDDTrain+; the validation split is not distinguished here.

| Subtype | Seen | Rows | Gate misses | Family errors | Correct R2L | Recall | Median binary score |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| guess_passwd | yes | 1231 | 1213 | 0 | 18 | 1.5% | 0.000 |
| warezmaster | yes | 944 | 549 | 5 | 390 | 41.3% | 0.066 |
| snmpguess | no | 331 | 329 | 2 | 0 | 0.0% | 0.000 |
| snmpgetattack | no | 178 | 173 | 5 | 0 | 0.0% | 0.000 |
| httptunnel | no | 133 | 114 | 19 | 0 | 0.0% | 0.035 |
| multihop | yes | 18 | 13 | 3 | 2 | 11.1% | 0.007 |
| named | no | 17 | 13 | 4 | 0 | 0.0% | 0.006 |
| sendmail | no | 14 | 13 | 0 | 1 | 7.1% | 0.008 |
| xlock | no | 9 | 8 | 1 | 0 | 0.0% | 0.003 |
| xsnoop | no | 4 | 4 | 0 | 0 | 0.0% | 0.003 |
| ftp_write | yes | 3 | 1 | 1 | 1 | 33.3% | 0.509 |
| phf | yes | 2 | 1 | 0 | 1 | 50.0% | 0.500 |
| imap | yes | 1 | 0 | 0 | 1 | 100.0% | 0.758 |

The companion JSON contains score quantiles by outcome and subtype, plus the attack-family choices after the gate. Scores are model outputs, not calibrated probabilities.

## Provenance

- `KDDTrain+.txt` SHA-256: `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95`
- `KDDTest+.txt` SHA-256: `fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84`
- `scaler` SHA-256: `32f0fe5af88eb644ceb01c30717bb56bdee97179d361af310140ba5f3ff9f0b4`
- `encoder` SHA-256: `975420bfdf7b4acbf70ac47722f412efa9b63e1d3931d0129e1cd0ae7a833ff2`
- `label_encoder` SHA-256: `8b6133e56b8fae55887cd489d8db3e6e805fe74466d625dcf65c149b05cc9b18`
- `metadata` SHA-256: `a1f5ef8809af80bd0858914952c2ba47ed9a74c76c9d1242687491f190b1a4f5`
- `binary_model` SHA-256: `4c01684bcda0a23a2c8fea855b95ee03082d92090405df9b39a89ad7af637db3`
- `multiclass_model` SHA-256: `a847da80ba8ff09d142d2160e6159baff63714eab5a2f54d216ea7c7e82f70eb`
