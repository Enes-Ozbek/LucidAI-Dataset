# LucidAI-Dataset report

## Main data (auto-1b-data)

| step | rows |
|---|---:|
| rows in the files | 725000 |
| dropped: user language is not English | -223222 |
| dropped: long-context rows (up to 65k tokens) | -17077 |
| dropped: not english | -6105 |
| dropped: exact duplicate | -1 |
| dropped: near duplicate | -7 |
| clean English rows | 478588 |
| **rows in train, val and test** | **478588** |

English check on the user requests, the test that decided each row: english function words 470654, english vocabulary 7381, detected es 3956, detected fr 1143, too little prose to check 561, detected de 399, non ascii text 347, detected pt 104, detected it 96, detected nl 12, detected tr 11, detected ca 10, detected pl 9, detected hu 4, detected hr 3, detected id 2, detected af 2, detected sq 2, detected sv 2, detected ro 1, detected fi 1, detected no 1.

## Splits

| split | rows | allow | deny | deny share | median chars A | median chars B | max chars B |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 335349 | 165541 | 169808 | 50.6% | 119 | 1044 | 9805 |
| val | 71626 | 36145 | 35481 | 49.5% | 133 | 1067 | 5734 |
| test | 71613 | 36102 | 35511 | 49.6% | 133 | 1067 | 8516 |
| ood_approve_or_deny | 1448 | 761 | 687 | 47.4% | 119 | 877 | 7410 |
| ood_saroku | 26633 | 12001 | 14632 | 54.9% | 135 | 499 | 885 |

Tool calls shared between splits (should be 0): train/val 0, train/test 0, val/test 0.
Calls that appear with both labels (only the user request or history decides): 9245 calls, 71805 rows.
Most repeated tool calls: 5898, 2118, 2008, 1985, 1520 rows.
Tool calls seen more than 20 times all went to train (`--train-only-calls`): 528 calls, 52503 rows.

Files: train: train_1.csv.gz + train_2.csv.gz + train_3.csv.gz; val: val.csv.gz; test: test.csv.gz; ood_approve_or_deny: ood_approve_or_deny.csv.gz; ood_saroku: ood_saroku.csv.gz.

## Difficulty

| difficulty | train | val | test |
|---|---:|---:|---:|
| easy | 122962 | 24128 | 24237 |
| medium | 114762 | 25714 | 25585 |
| hard | 97625 | 21784 | 21791 |

## Length bucket (how much agent history)

| length_bucket | train | val | test |
|---|---:|---:|---:|
| short | 112547 | 24312 | 24387 |
| long | 111787 | 23604 | 23526 |
| medium | 111015 | 23710 | 23700 |

## Agent framework

| framework | train | val | test |
|---|---:|---:|---:|
| computer_use | 22701 | 2490 | 2409 |
| ci_agent | 19633 | 4757 | 4723 |
| home_iot | 19437 | 3710 | 3767 |
| langchain | 19333 | 3833 | 3818 |
| pkg_agent | 19300 | 3956 | 3968 |
| mcp_postgres | 19175 | 4242 | 4308 |
| finance_agent | 19130 | 3894 | 3887 |
| codex_cli | 19048 | 3772 | 3858 |
| email_calendar | 18821 | 3958 | 4059 |
| devops_agent | 18557 | 4613 | 4497 |
| claude_code | 18446 | 3893 | 3806 |
| data_agent | 18290 | 4301 | 4326 |
| cursor_agent | 18282 | 4098 | 3963 |
| mcp_browser | 18280 | 4397 | 4577 |
| mcp_slack | 18180 | 4255 | 4384 |
| openai_functions | 18085 | 3773 | 3768 |
| mcp_filesystem | 16829 | 4232 | 4001 |
| mcp_github | 13822 | 3452 | 3494 |

## Categories (each one gives the label away: error analysis only)

| category | train | val | test |
|---|---:|---:|---:|
| routine_benign | 20111 | 4734 | 4663 |
| explicitly_authorized | 19675 | 4547 | 4502 |
| destructive_filesystem | 19282 | 3433 | 3453 |
| scoped_destructive | 19146 | 4156 | 4222 |
| privileged_but_normal | 18978 | 3340 | 3288 |
| scary_but_scoped | 18949 | 4216 | 3994 |
| network_is_fine | 18881 | 4581 | 4784 |
| read_only_sensitive | 16668 | 2855 | 2782 |
| legit_remote_write | 16602 | 3919 | 4039 |
| recovery_action | 16531 | 3797 | 3828 |
| data_destruction_unrequested | 15912 | 2357 | 2330 |
| irreversible_remote | 15295 | 3029 | 3049 |
| scope_violation | 14580 | 2676 | 2684 |
| abuse_spam_privacy | 12409 | 2942 | 2953 |
| malicious_install | 11531 | 2512 | 2604 |
| physical_financial | 11436 | 2605 | 2570 |
| privilege_escalation | 10969 | 1944 | 1919 |
| obfuscated_payload | 10357 | 2463 | 2514 |
| supply_chain | 10162 | 2513 | 2423 |
| covering_tracks | 9886 | 2103 | 2113 |
| credential_exfiltration | 9662 | 2468 | 2386 |
| prompt_injection | 9476 | 2174 | 2305 |
| self_permission_tamper | 8851 | 2262 | 2208 |

## Outside test: approve-or-deny benchmark

3000 rows in the files, not english lang -896, long context -367, not english -31, call already in the main data -258.

## Outside test: saroku

26731 rows in the files, near duplicate -60, call already in the main data -38.
