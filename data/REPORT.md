# MayI dataset report

## Main data (auto-1b-data)

| step | rows |
|---|---:|
| rows in the files | 725000 |
| dropped: user language is not English | -223222 |
| dropped: long-context rows (up to 65k tokens) | -17077 |
| dropped: not in the short length bucket | -321251 |
| dropped: not english | -2430 |
| dropped: call too long | -504 |
| dropped: input too long | -419 |
| dropped: exact duplicate | -1 |
| dropped: near duplicate | -4 |
| clean English short rows | 160092 |
| sampled for the splits (`--max-rows`) | 58516 |
| dropped: extra rows of a tool call seen more than 20 times (`--max-per-call`) | -2260 |
| **rows in train, val and test** | **56256** |

English check on the user requests, the test that decided each row: english function words 158054, english vocabulary 2681, detected es 1547, detected fr 544, too little prose to check 285, detected de 138, non ascii text 92, detected it 40, detected pt 40, detected ca 7, detected tr 5, detected nl 5, detected pl 4, detected hr 3, detected sv 2, detected af 1, detected hu 1, detected id 1.

## Splits

| split | rows | allow | deny | deny share | median chars A | median chars B | max chars B |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 39519 | 19504 | 20015 | 50.6% | 131 | 597 | 1200 |
| val | 8382 | 4144 | 4238 | 50.6% | 133 | 601 | 1196 |
| test | 8355 | 4191 | 4164 | 49.8% | 132 | 599 | 1199 |
| ood_approve_or_deny | 609 | 347 | 262 | 43.0% | 116 | 553 | 1138 |
| ood_saroku | 3023 | 1350 | 1673 | 55.3% | 135 | 498 | 821 |

Tool calls shared between splits (should be 0): train/val 0, train/test 0, val/test 0.
Calls that appear with both labels (only the user request or history decides): 1356 calls, 4539 rows.
Most repeated tool calls before the cap: 597, 181, 171, 163, 144 rows; 47 calls were cut to 20 rows.

## Difficulty

| difficulty | train | val | test |
|---|---:|---:|---:|
| easy | 14298 | 3005 | 3015 |
| medium | 13478 | 2876 | 2873 |
| hard | 11743 | 2501 | 2467 |

## Agent framework

| framework | train | val | test |
|---|---:|---:|---:|
| ci_agent | 2539 | 536 | 568 |
| devops_agent | 2531 | 535 | 482 |
| mcp_postgres | 2516 | 532 | 515 |
| mcp_slack | 2426 | 510 | 482 |
| email_calendar | 2365 | 481 | 476 |
| mcp_browser | 2281 | 485 | 529 |
| cursor_agent | 2260 | 426 | 457 |
| data_agent | 2247 | 530 | 478 |
| langchain | 2245 | 467 | 472 |
| codex_cli | 2204 | 482 | 445 |
| pkg_agent | 2191 | 393 | 434 |
| claude_code | 2157 | 495 | 470 |
| finance_agent | 2079 | 477 | 477 |
| home_iot | 2066 | 450 | 418 |
| mcp_filesystem | 2044 | 440 | 489 |
| openai_functions | 1954 | 405 | 403 |
| mcp_github | 1847 | 404 | 426 |
| computer_use | 1567 | 334 | 334 |

## Categories (each one gives the label away: error analysis only)

| category | train | val | test |
|---|---:|---:|---:|
| explicitly_authorized | 2449 | 517 | 513 |
| routine_benign | 2419 | 524 | 519 |
| network_is_fine | 2365 | 491 | 523 |
| scary_but_scoped | 2259 | 492 | 458 |
| scoped_destructive | 2242 | 477 | 501 |
| legit_remote_write | 2029 | 421 | 439 |
| recovery_action | 2010 | 437 | 421 |
| destructive_filesystem | 2009 | 435 | 432 |
| privileged_but_normal | 1986 | 404 | 378 |
| read_only_sensitive | 1745 | 381 | 439 |
| irreversible_remote | 1717 | 367 | 349 |
| abuse_spam_privacy | 1580 | 338 | 328 |
| scope_violation | 1541 | 351 | 306 |
| data_destruction_unrequested | 1484 | 272 | 283 |
| malicious_install | 1436 | 271 | 306 |
| physical_financial | 1373 | 306 | 280 |
| supply_chain | 1357 | 249 | 279 |
| obfuscated_payload | 1333 | 327 | 295 |
| credential_exfiltration | 1315 | 283 | 258 |
| privilege_escalation | 1274 | 273 | 276 |
| covering_tracks | 1263 | 254 | 267 |
| prompt_injection | 1229 | 265 | 251 |
| self_permission_tamper | 1104 | 247 | 254 |

## Outside test: approve-or-deny benchmark

3000 rows in the files, not english lang -896, long context -367, not short bucket -1077, not english -12, call too long -7, input too long -8, call already in the main data -24.

## Outside test: saroku

26731 rows in the files, near duplicate -60, call already in the main data -3, sampled 3023.
