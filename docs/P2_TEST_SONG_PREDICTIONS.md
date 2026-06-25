# P2 Test Song Prediction Details

Generated: 2026-06-25

Notes: `rank=1` means the true producer is the model Top-1. `Rejected` means the model returned candidates, but the confidence is below the calibrated acceptance threshold.

## Summary

| Split | Songs | Top-1 | Top-3 | Macro-F1 | MRR | Coverage | Accepted accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Final frozen test | 124 | 78.23% | 92.74% | 78.24% | 86.02% | 64.52% | 96.25% |
| Development holdout | 62 | 75.81% | 80.65% | 74.42% | 80.32% | 69.35% | 86.05% |

## Final frozen test: Top-1 errors

| # | True producer | Song | Model Top-1 | True rank | Top-1 | Accept state | Confidence | Top-3 |
|---:|---|---|---|---:|---|---|---:|---|
| 1 | 40mP | Initial Song | すりぃ | 2 | Wrong | Rejected | 39.62% | すりぃ 39.62% / 40mP 32.02% / じん 14.68% |
| 2 | 40mP | タイムマシン | n-buna | 2 | Wrong | Rejected | 47.16% | n-buna 47.16% / 40mP 46.10% / Neru 2.17% |
| 3 | Ayase | 先天性アサルトガール | cosMo@暴走P | 3 | Wrong | Rejected | 33.15% | cosMo@暴走P 33.15% / じん 18.42% / Ayase 13.53% |
| 4 | Ayase | 怪物 | DECO*27 | 6 | Wrong | Rejected | 30.99% | DECO*27 30.99% / すりぃ 30.55% / じん 11.65% |
| 5 | DECO*27 | ヒバナ | Neru | 2 | Wrong | Rejected | 41.92% | Neru 41.92% / DECO*27 21.60% / すりぃ 21.16% |
| 6 | DECO*27 | 弱虫モンブラン | 40mP | 6 | Wrong | Accepted | 80.64% | 40mP 80.64% / すりぃ 3.58% / はるまきごはん 2.38% |
| 7 | DECO*27 | 愛言葉 | R Sound Design | 3 | Wrong | Rejected | 25.22% | R Sound Design 25.22% / 40mP 18.12% / DECO*27 11.95% |
| 8 | Kanaria | エッサホイサ | じん | 17 | Wrong | Rejected | 20.09% | じん 20.09% / すりぃ 19.54% / sasakure.UK 10.76% |
| 9 | kemu | ラットホープ | r-906 | 2 | Wrong | Rejected | 47.10% | r-906 47.10% / kemu 23.49% / とあ 14.61% |
| 10 | kemu | 十年越しのラストピース | cosMo@暴走P | 2 | Wrong | Rejected | 28.41% | cosMo@暴走P 28.41% / kemu 25.74% / じん 18.87% |
| 11 | Neru | い〜やい〜やい〜や | すりぃ | 2 | Wrong | Rejected | 30.40% | すりぃ 30.40% / Neru 24.82% / じん 14.90% |
| 12 | Neru | 病名は愛だった | すりぃ | 2 | Wrong | Rejected | 35.63% | すりぃ 35.63% / Neru 34.15% / sasakure.UK 15.46% |
| 13 | R Sound Design | マンダリン | すりぃ | 24 | Wrong | Rejected | 19.87% | すりぃ 19.87% / Chinozo 14.63% / ピノキオピー 14.56% |
| 14 | r-906 | スーパーノヴァ | Neru | 15 | Wrong | Rejected | 56.48% | Neru 56.48% / すりぃ 12.19% / じん 8.50% |
| 15 | sasakure.UK | 化孵化 | cosMo@暴走P | 2 | Wrong | Rejected | 51.60% | cosMo@暴走P 51.60% / sasakure.UK 23.36% / いよわ 8.31% |
| 16 | wowaka | 白塔 | とあ | 15 | Wrong | Rejected | 49.11% | とあ 49.11% / r-906 31.14% / なきそ 9.78% |
| 17 | すりぃ | めめしぃ | とあ | 4 | Wrong | Accepted | 86.65% | とあ 86.65% / 40mP 7.29% / てにをは 2.26% |
| 18 | すりぃ | 空中分解 | じん | 2 | Wrong | Rejected | 41.19% | じん 41.19% / すりぃ 32.72% / DECO*27 15.50% |
| 19 | てにをは | BO-ZU | じん | 12 | Wrong | Rejected | 26.99% | じん 26.99% / r-906 13.56% / sasakure.UK 8.63% |
| 20 | てにをは | オノマトペ天使 | じん | 2 | Wrong | Rejected | 36.45% | じん 36.45% / てにをは 24.31% / 煮ル果実 6.41% |
| 21 | なきそ | 少女失格 | じん | 10 | Wrong | Rejected | 43.09% | じん 43.09% / cosMo@暴走P 16.58% / はるまきごはん 14.69% |
| 22 | なきそ | 花めかない | とあ | 2 | Wrong | Rejected | 25.72% | とあ 25.72% / なきそ 23.92% / sasakure.UK 13.89% |
| 23 | はるまきごはん | ぽかぽかの星 | Neru | 2 | Wrong | Rejected | 36.59% | Neru 36.59% / はるまきごはん 36.42% / すりぃ 11.42% |
| 24 | はるまきごはん | セブンティーナ | sasakure.UK | 2 | Wrong | Accepted | 81.10% | sasakure.UK 81.10% / はるまきごはん 11.31% / cosMo@暴走P 2.48% |
| 25 | はるまきごはん | 銀河録 | じん | 2 | Wrong | Rejected | 57.10% | じん 57.10% / はるまきごはん 20.99% / 40mP 6.00% |
| 26 | ピノキオピー | すきなことだけでいいです | cosMo@暴走P | 2 | Wrong | Rejected | 42.84% | cosMo@暴走P 42.84% / ピノキオピー 13.32% / sasakure.UK 8.20% |
| 27 | ピノキオピー | ぼくらはみんな意味不明 | DECO*27 | 2 | Wrong | Rejected | 25.70% | DECO*27 25.70% / ピノキオピー 20.96% / すりぃ 14.52% |

## Development holdout: Top-1 errors

| # | True producer | Song | Model Top-1 | True rank | Top-1 | Accept state | Confidence | Top-3 |
|---:|---|---|---|---:|---|---|---:|---|
| 1 | Ayase | アイドル | DECO*27 | 17 | Wrong | Accepted | 83.10% | DECO*27 83.10% / じん 6.83% / すりぃ 4.61% |
| 2 | DECO*27 | 二息歩行 | sasakure.UK | 2 | Wrong | Rejected | 22.63% | sasakure.UK 22.63% / DECO*27 17.58% / じん 11.08% |
| 3 | Kanaria | チャンピオン | cosMo@暴走P | 7 | Wrong | Rejected | 21.02% | cosMo@暴走P 21.02% / DECO*27 15.89% / すりぃ 13.26% |
| 4 | kemu | 88☆彡 | DECO*27 | 8 | Wrong | Rejected | 47.93% | DECO*27 47.93% / すりぃ 23.73% / はるまきごはん 7.83% |
| 5 | kemu | セカイ | 40mP | 20 | Wrong | Rejected | 34.64% | 40mP 34.64% / DECO*27 32.18% / じん 17.56% |
| 6 | n-buna | 言って。 | じん | 2 | Wrong | Accepted | 63.13% | じん 63.13% / n-buna 20.22% / Neru 9.06% |
| 7 | Orangestar | スターナイトスノウ | n-buna | 5 | Wrong | Rejected | 58.67% | n-buna 58.67% / Neru 25.19% / すりぃ 6.61% |
| 8 | syudou | うっせぇわ | すりぃ | 5 | Wrong | Accepted | 87.03% | すりぃ 87.03% / てにをは 2.15% / Chinozo 1.78% |
| 9 | wowaka | and I'm home | とあ | 29 | Wrong | Rejected | 33.04% | とあ 33.04% / じん 20.25% / すりぃ 11.50% |
| 10 | すりぃ | メンヘラ取扱説明書 | kemu | 7 | Wrong | Accepted | 67.89% | kemu 67.89% / DECO*27 11.42% / じん 8.42% |
| 11 | てにをは | 名探偵連続殺人事件 | 煮ル果実 | 7 | Wrong | Rejected | 22.28% | 煮ル果実 22.28% / DECO*27 15.34% / すりぃ 12.00% |
| 12 | なきそ | 惡ふざけ | cosMo@暴走P | 2 | Wrong | Rejected | 32.42% | cosMo@暴走P 32.42% / なきそ 14.55% / じん 14.01% |
| 13 | ナユタン星人 | オペラ！スペースオペラ！  | cosMo@暴走P | 28 | Wrong | Accepted | 67.63% | cosMo@暴走P 67.63% / じん 12.29% / DECO*27 11.22% |
| 14 | ナユタン星人 | モア！ジャンプ！モア！ | すりぃ | 12 | Wrong | Rejected | 32.72% | すりぃ 32.72% / Neru 22.71% / n-buna 19.20% |
| 15 | ハチ | KICK BACK | じん | 12 | Wrong | Accepted | 67.78% | じん 67.78% / すりぃ 12.70% / DECO*27 12.38% |

## Final frozen test: all 124 songs

| # | True producer | Song | Model Top-1 | True rank | Top-1 | Accept state | Confidence | Top-3 |
|---:|---|---|---|---:|---|---|---:|---|
| 1 | 40mP | Initial Song | すりぃ | 2 | Wrong | Rejected | 39.62% | すりぃ 39.62% / 40mP 32.02% / じん 14.68% |
| 2 | 40mP | Warning! | 40mP | 1 | Correct | Accepted | 72.94% | 40mP 72.94% / すりぃ 9.50% / Chinozo 7.09% |
| 3 | 40mP | だんだん早くなる | 40mP | 1 | Correct | Accepted | 99.59% | 40mP 99.59% / とあ 0.19% / Neru 0.08% |
| 4 | 40mP | タイムマシン | n-buna | 2 | Wrong | Rejected | 47.16% | n-buna 47.16% / 40mP 46.10% / Neru 2.17% |
| 5 | Ayase | ヴァイオレッタ | Ayase | 1 | Correct | Accepted | 96.44% | Ayase 96.44% / はるまきごはん 0.87% / すりぃ 0.71% |
| 6 | Ayase | 先天性アサルトガール | cosMo@暴走P | 3 | Wrong | Rejected | 33.15% | cosMo@暴走P 33.15% / じん 18.42% / Ayase 13.53% |
| 7 | Ayase | 怪物 | DECO*27 | 6 | Wrong | Rejected | 30.99% | DECO*27 30.99% / すりぃ 30.55% / じん 11.65% |
| 8 | Ayase | 泣いてない | Ayase | 1 | Correct | Accepted | 89.11% | Ayase 89.11% / とあ 2.87% / はるまきごはん 2.45% |
| 9 | Chinozo | TAMAYA | Chinozo | 1 | Correct | Accepted | 80.65% | Chinozo 80.65% / すりぃ 16.47% / てにをは 1.77% |
| 10 | Chinozo | だまってちゃん | Chinozo | 1 | Correct | Accepted | 65.65% | Chinozo 65.65% / DECO*27 13.04% / てにをは 9.08% |
| 11 | Chinozo | ミィハー | Chinozo | 1 | Correct | Rejected | 51.03% | Chinozo 51.03% / DECO*27 11.45% / ピノキオピー 10.30% |
| 12 | Chinozo | ムシ | Chinozo | 1 | Correct | Rejected | 40.89% | Chinozo 40.89% / Neru 27.57% / てにをは 11.45% |
| 13 | cosMo@暴走P | #誰かこの痛みに名前をつけてください | cosMo@暴走P | 1 | Correct | Accepted | 68.60% | cosMo@暴走P 68.60% / kemu 10.56% / すりぃ 6.58% |
| 14 | cosMo@暴走P | Anti the EuphoriaHOLiC | cosMo@暴走P | 1 | Correct | Accepted | 98.98% | cosMo@暴走P 98.98% / ハチ 0.42% / いよわ 0.11% |
| 15 | cosMo@暴走P | ウシノヒ☆アブダクション | cosMo@暴走P | 1 | Correct | Accepted | 97.33% | cosMo@暴走P 97.33% / MARETU 1.22% / Mitchie M 0.63% |
| 16 | cosMo@暴走P | キノコがはえてる!! | cosMo@暴走P | 1 | Correct | Accepted | 93.92% | cosMo@暴走P 93.92% / MARETU 2.19% / sasakure.UK 1.43% |
| 17 | DECO*27 | ストリーミングハート | DECO*27 | 1 | Correct | Rejected | 53.00% | DECO*27 53.00% / じん 11.54% / cosMo@暴走P 9.92% |
| 18 | DECO*27 | ヒバナ | Neru | 2 | Wrong | Rejected | 41.92% | Neru 41.92% / DECO*27 21.60% / すりぃ 21.16% |
| 19 | DECO*27 | 弱虫モンブラン | 40mP | 6 | Wrong | Accepted | 80.64% | 40mP 80.64% / すりぃ 3.58% / はるまきごはん 2.38% |
| 20 | DECO*27 | 愛言葉 | R Sound Design | 3 | Wrong | Rejected | 25.22% | R Sound Design 25.22% / 40mP 18.12% / DECO*27 11.95% |
| 21 | Kanaria | Dec. | Kanaria | 1 | Correct | Accepted | 97.73% | Kanaria 97.73% / DECO*27 0.65% / Chinozo 0.56% |
| 22 | Kanaria | アイロニック | Kanaria | 1 | Correct | Rejected | 58.24% | Kanaria 58.24% / すりぃ 8.18% / Ayase 7.29% |
| 23 | Kanaria | エッサホイサ | じん | 17 | Wrong | Rejected | 20.09% | じん 20.09% / すりぃ 19.54% / sasakure.UK 10.76% |
| 24 | Kanaria | デーモンロード | Kanaria | 1 | Correct | Accepted | 97.29% | Kanaria 97.29% / ハチ 0.71% / ピノキオピー 0.48% |
| 25 | kemu | Buzzing | kemu | 1 | Correct | Accepted | 66.67% | kemu 66.67% / MARETU 10.23% / n-buna 5.41% |
| 26 | kemu | ラットホープ | r-906 | 2 | Wrong | Rejected | 47.10% | r-906 47.10% / kemu 23.49% / とあ 14.61% |
| 27 | kemu | 十年越しのラストピース | cosMo@暴走P | 2 | Wrong | Rejected | 28.41% | cosMo@暴走P 28.41% / kemu 25.74% / じん 18.87% |
| 28 | kemu | 花呼ぶ声 | kemu | 1 | Correct | Rejected | 37.73% | kemu 37.73% / じん 30.52% / DECO*27 22.10% |
| 29 | MARETU | うみたがり | MARETU | 1 | Correct | Accepted | 76.71% | MARETU 76.71% / すりぃ 6.93% / Neru 6.09% |
| 30 | MARETU | しう | MARETU | 1 | Correct | Accepted | 99.45% | MARETU 99.45% / すりぃ 0.16% / じん 0.12% |
| 31 | MARETU | ゴキブリの味 | MARETU | 1 | Correct | Accepted | 99.45% | MARETU 99.45% / sasakure.UK 0.11% / すりぃ 0.10% |
| 32 | MARETU | ダーリン | MARETU | 1 | Correct | Accepted | 95.87% | MARETU 95.87% / すりぃ 1.30% / じん 0.97% |
| 33 | MIMI | だきしめるまで。 | MIMI | 1 | Correct | Accepted | 76.64% | MIMI 76.64% / とあ 21.36% / sasakure.UK 0.75% |
| 34 | MIMI | はぐ | MIMI | 1 | Correct | Accepted | 99.46% | MIMI 99.46% / すりぃ 0.20% / Ayase 0.17% |
| 35 | MIMI | もーいいかい | MIMI | 1 | Correct | Accepted | 97.10% | MIMI 97.10% / Ayase 0.97% / すりぃ 0.81% |
| 36 | MIMI | ヒミツ | MIMI | 1 | Correct | Accepted | 99.19% | MIMI 99.19% / Ayase 0.35% / すりぃ 0.23% |
| 37 | Mitchie M | アイドルを咲かせ | Mitchie M | 1 | Correct | Accepted | 99.75% | Mitchie M 99.75% / MARETU 0.14% / かいりきベア 0.06% |
| 38 | Mitchie M | 好き！雪！本気マジック | Mitchie M | 1 | Correct | Accepted | 99.87% | Mitchie M 99.87% / 煮ル果実 0.02% / DECO*27 0.02% |
| 39 | Mitchie M | 徳川カップヌードル禁止令 | Mitchie M | 1 | Correct | Accepted | 95.16% | Mitchie M 95.16% / ハチ 0.80% / かいりきベア 0.65% |
| 40 | Mitchie M | 愛Dee | Mitchie M | 1 | Correct | Accepted | 99.99% | Mitchie M 99.99% / MARETU 0.00% / DECO*27 0.00% |
| 41 | n-buna | さよならワンダーノイズ | n-buna | 1 | Correct | Accepted | 84.85% | n-buna 84.85% / じん 6.56% / Neru 2.32% |
| 42 | n-buna | ただ君に晴れ | n-buna | 1 | Correct | Rejected | 36.37% | n-buna 36.37% / Neru 34.44% / じん 14.26% |
| 43 | n-buna | もうじき夏が終わるから | n-buna | 1 | Correct | Rejected | 46.62% | n-buna 46.62% / Orangestar 10.77% / Neru 10.48% |
| 44 | n-buna | 背景、夏に溺れる | n-buna | 1 | Correct | Accepted | 87.59% | n-buna 87.59% / じん 6.43% / Neru 3.13% |
| 45 | Neru | FPS | Neru | 1 | Correct | Accepted | 85.90% | Neru 85.90% / すりぃ 6.91% / じん 3.70% |
| 46 | Neru | い〜やい〜やい〜や | すりぃ | 2 | Wrong | Rejected | 30.40% | すりぃ 30.40% / Neru 24.82% / じん 14.90% |
| 47 | Neru | 捨て子のステラ | Neru | 1 | Correct | Accepted | 98.04% | Neru 98.04% / すりぃ 0.58% / てにをは 0.35% |
| 48 | Neru | 病名は愛だった | すりぃ | 2 | Wrong | Rejected | 35.63% | すりぃ 35.63% / Neru 34.15% / sasakure.UK 15.46% |
| 49 | Orangestar | Encounter | Orangestar | 1 | Correct | Accepted | 77.77% | Orangestar 77.77% / sasakure.UK 5.03% / はるまきごはん 3.64% |
| 50 | Orangestar | 濫觴生命 | Orangestar | 1 | Correct | Accepted | 99.45% | Orangestar 99.45% / すりぃ 0.13% / 40mP 0.09% |
| 51 | Orangestar | 雨き声残響 | Orangestar | 1 | Correct | Rejected | 33.73% | Orangestar 33.73% / じん 20.70% / すりぃ 16.34% |
| 52 | Orangestar | 霽れを待つ | Orangestar | 1 | Correct | Accepted | 87.55% | Orangestar 87.55% / R Sound Design 2.71% / Neru 1.60% |
| 53 | R Sound Design | Nightscape | R Sound Design | 1 | Correct | Accepted | 99.37% | R Sound Design 99.37% / sasakure.UK 0.14% / なきそ 0.08% |
| 54 | R Sound Design | マンダリン | すりぃ | 24 | Wrong | Rejected | 19.87% | すりぃ 19.87% / Chinozo 14.63% / ピノキオピー 14.56% |
| 55 | R Sound Design | 夜と幽霊 | R Sound Design | 1 | Correct | Accepted | 98.10% | R Sound Design 98.10% / Neru 0.44% / すりぃ 0.41% |
| 56 | R Sound Design | 水星都市計画 | R Sound Design | 1 | Correct | Accepted | 77.26% | R Sound Design 77.26% / とあ 5.64% / sasakure.UK 5.17% |
| 57 | r-906 | スーパーノヴァ | Neru | 15 | Wrong | Rejected | 56.48% | Neru 56.48% / すりぃ 12.19% / じん 8.50% |
| 58 | r-906 | ユメミ | r-906 | 1 | Correct | Rejected | 60.11% | r-906 60.11% / すりぃ 12.59% / R Sound Design 8.51% |
| 59 | r-906 | 怪電話 | r-906 | 1 | Correct | Accepted | 98.73% | r-906 98.73% / cosMo@暴走P 0.75% / sasakure.UK 0.31% |
| 60 | r-906 | 梅に鶯 | r-906 | 1 | Correct | Accepted | 68.55% | r-906 68.55% / いよわ 15.94% / 稲葉曇 2.88% |
| 61 | sasakure.UK | Snow Song Show | sasakure.UK | 1 | Correct | Rejected | 58.85% | sasakure.UK 58.85% / ハチ 19.89% / R Sound Design 12.64% |
| 62 | sasakure.UK | タイガーランペイジ | sasakure.UK | 1 | Correct | Accepted | 94.43% | sasakure.UK 94.43% / cosMo@暴走P 2.18% / r-906 0.76% |
| 63 | sasakure.UK | ポジネガ＊ミステイカーズ | sasakure.UK | 1 | Correct | Accepted | 99.93% | sasakure.UK 99.93% / cosMo@暴走P 0.02% / はるまきごはん 0.02% |
| 64 | sasakure.UK | 化孵化 | cosMo@暴走P | 2 | Wrong | Rejected | 51.60% | cosMo@暴走P 51.60% / sasakure.UK 23.36% / いよわ 8.31% |
| 65 | syudou | へべれけジャンキー | syudou | 1 | Correct | Accepted | 99.72% | syudou 99.72% / すりぃ 0.09% / てにをは 0.05% |
| 66 | syudou | キャラバン | syudou | 1 | Correct | Accepted | 99.99% | syudou 99.99% / すりぃ 0.01% / ピノキオピー 0.00% |
| 67 | syudou | ボニータ | syudou | 1 | Correct | Accepted | 99.96% | syudou 99.96% / かいりきベア 0.01% / wowaka 0.00% |
| 68 | syudou | 爆笑 | syudou | 1 | Correct | Accepted | 99.06% | syudou 99.06% / ピノキオピー 0.37% / はるまきごはん 0.12% |
| 69 | wowaka | テノヒラ | wowaka | 1 | Correct | Accepted | 88.88% | wowaka 88.88% / すりぃ 5.76% / 40mP 1.41% |
| 70 | wowaka | ラインアート | wowaka | 1 | Correct | Accepted | 99.92% | wowaka 99.92% / kemu 0.03% / cosMo@暴走P 0.02% |
| 71 | wowaka | リバシブルドール | wowaka | 1 | Correct | Accepted | 79.62% | wowaka 79.62% / かいりきベア 6.41% / DECO*27 3.10% |
| 72 | wowaka | 白塔 | とあ | 15 | Wrong | Rejected | 49.11% | とあ 49.11% / r-906 31.14% / なきそ 9.78% |
| 73 | いよわ | あだぽしゃ | いよわ | 1 | Correct | Accepted | 99.95% | いよわ 99.95% / はるまきごはん 0.02% / cosMo@暴走P 0.01% |
| 74 | いよわ | パジャミィ | いよわ | 1 | Correct | Accepted | 93.78% | いよわ 93.78% / じん 2.28% / DECO*27 1.54% |
| 75 | いよわ | 大女優さん | いよわ | 1 | Correct | Accepted | 93.70% | いよわ 93.70% / sasakure.UK 1.68% / cosMo@暴走P 1.58% |
| 76 | いよわ | 黄金数 | いよわ | 1 | Correct | Accepted | 99.47% | いよわ 99.47% / sasakure.UK 0.24% / cosMo@暴走P 0.15% |
| 77 | かいりきベア | セイデンキニンゲン | かいりきベア | 1 | Correct | Rejected | 62.03% | かいりきベア 62.03% / n-buna 32.25% / じん 1.82% |
| 78 | かいりきベア | ネロイズム | かいりきベア | 1 | Correct | Accepted | 98.89% | かいりきベア 98.89% / Neru 0.39% / Mitchie M 0.24% |
| 79 | かいりきベア | マネマネサイコトロピック | かいりきベア | 1 | Correct | Rejected | 24.29% | かいりきベア 24.29% / DECO*27 17.87% / 40mP 15.33% |
| 80 | かいりきベア | 病み垢ステロイド | かいりきベア | 1 | Correct | Accepted | 97.70% | かいりきベア 97.70% / すりぃ 0.71% / DECO*27 0.36% |
| 81 | じん | アヤノの幸福理論 | じん | 1 | Correct | Rejected | 55.42% | じん 55.42% / すりぃ 17.19% / 40mP 16.80% |
| 82 | じん | オツキミリサイタル | じん | 1 | Correct | Accepted | 92.30% | じん 92.30% / Neru 3.39% / すりぃ 2.74% |
| 83 | じん | ロスタイムメモリー | じん | 1 | Correct | Accepted | 99.36% | じん 99.36% / Neru 0.32% / すりぃ 0.16% |
| 84 | じん | 日本橋高架下R計画 | じん | 1 | Correct | Rejected | 43.98% | じん 43.98% / Neru 21.54% / Chinozo 10.07% |
| 85 | すりぃ | めめしぃ | とあ | 4 | Wrong | Accepted | 86.65% | とあ 86.65% / 40mP 7.29% / てにをは 2.26% |
| 86 | すりぃ | アンビバレンス | すりぃ | 1 | Correct | Rejected | 58.01% | すりぃ 58.01% / DECO*27 21.49% / じん 8.12% |
| 87 | すりぃ | モーニングループ | すりぃ | 1 | Correct | Rejected | 27.77% | すりぃ 27.77% / とあ 26.76% / じん 13.18% |
| 88 | すりぃ | 空中分解 | じん | 2 | Wrong | Rejected | 41.19% | じん 41.19% / すりぃ 32.72% / DECO*27 15.50% |
| 89 | てにをは | BO-ZU | じん | 12 | Wrong | Rejected | 26.99% | じん 26.99% / r-906 13.56% / sasakure.UK 8.63% |
| 90 | てにをは | オノマトペ天使 | じん | 2 | Wrong | Rejected | 36.45% | じん 36.45% / てにをは 24.31% / 煮ル果実 6.41% |
| 91 | てにをは | クーロンズ・ホテル | てにをは | 1 | Correct | Rejected | 32.35% | てにをは 32.35% / cosMo@暴走P 18.47% / はるまきごはん 9.32% |
| 92 | てにをは | トレンドキラー | てにをは | 1 | Correct | Accepted | 96.04% | てにをは 96.04% / DECO*27 0.95% / Chinozo 0.82% |
| 93 | とあ | M | とあ | 1 | Correct | Rejected | 31.04% | とあ 31.04% / cosMo@暴走P 24.18% / Ayase 14.40% |
| 94 | とあ | ドライドライフラワー | とあ | 1 | Correct | Accepted | 99.45% | とあ 99.45% / すりぃ 0.13% / てにをは 0.08% |
| 95 | とあ | ユメハミ | とあ | 1 | Correct | Accepted | 91.66% | とあ 91.66% / MIMI 4.50% / すりぃ 1.76% |
| 96 | とあ | 恋の才能 | とあ | 1 | Correct | Accepted | 95.58% | とあ 95.58% / じん 1.15% / なきそ 0.96% |
| 97 | なきそ | シロガラス | なきそ | 1 | Correct | Rejected | 43.94% | なきそ 43.94% / はるまきごはん 30.00% / cosMo@暴走P 8.88% |
| 98 | なきそ | 少女失格 | じん | 10 | Wrong | Rejected | 43.09% | じん 43.09% / cosMo@暴走P 16.58% / はるまきごはん 14.69% |
| 99 | なきそ | 甘ったる   | なきそ | 1 | Correct | Accepted | 99.90% | なきそ 99.90% / てにをは 0.03% / じん 0.02% |
| 100 | なきそ | 花めかない | とあ | 2 | Wrong | Rejected | 25.72% | とあ 25.72% / なきそ 23.92% / sasakure.UK 13.89% |
| 101 | はるまきごはん | ぽかぽかの星 | Neru | 2 | Wrong | Rejected | 36.59% | Neru 36.59% / はるまきごはん 36.42% / すりぃ 11.42% |
| 102 | はるまきごはん | セブンティーナ | sasakure.UK | 2 | Wrong | Accepted | 81.10% | sasakure.UK 81.10% / はるまきごはん 11.31% / cosMo@暴走P 2.48% |
| 103 | はるまきごはん | 再会 | はるまきごはん | 1 | Correct | Accepted | 98.78% | はるまきごはん 98.78% / sasakure.UK 0.29% / すりぃ 0.24% |
| 104 | はるまきごはん | 銀河録 | じん | 2 | Wrong | Rejected | 57.10% | じん 57.10% / はるまきごはん 20.99% / 40mP 6.00% |
| 105 | ナユタン星人 | アンドロメダアンドロメダ | ナユタン星人 | 1 | Correct | Accepted | 98.59% | ナユタン星人 98.59% / かいりきベア 0.50% / MARETU 0.16% |
| 106 | ナユタン星人 | ハウトゥワープ | ナユタン星人 | 1 | Correct | Accepted | 99.92% | ナユタン星人 99.92% / かいりきベア 0.04% / DECO*27 0.01% |
| 107 | ナユタン星人 | パーフェクト生命 | ナユタン星人 | 1 | Correct | Accepted | 99.94% | ナユタン星人 99.94% / Neru 0.01% / じん 0.01% |
| 108 | ナユタン星人 | 飛行少女 | ナユタン星人 | 1 | Correct | Accepted | 99.95% | ナユタン星人 99.95% / かいりきベア 0.03% / DECO*27 0.00% |
| 109 | ハチ | リンネ | ハチ | 1 | Correct | Accepted | 87.40% | ハチ 87.40% / じん 1.86% / cosMo@暴走P 1.81% |
| 110 | ハチ | 恋人のランジェ  | ハチ | 1 | Correct | Accepted | 98.68% | ハチ 98.68% / 40mP 0.43% / てにをは 0.21% |
| 111 | ハチ | 演劇テレプシコーラ | ハチ | 1 | Correct | Accepted | 99.93% | ハチ 99.93% / cosMo@暴走P 0.02% / じん 0.01% |
| 112 | ハチ | 白痴 | ハチ | 1 | Correct | Accepted | 94.50% | ハチ 94.50% / sasakure.UK 1.65% / 煮ル果実 1.58% |
| 113 | ピノキオピー | ありふれたせかいせいふく | ピノキオピー | 1 | Correct | Rejected | 48.96% | ピノキオピー 48.96% / はるまきごはん 29.33% / DECO*27 9.10% |
| 114 | ピノキオピー | きみも悪い人でよかった | ピノキオピー | 1 | Correct | Accepted | 63.99% | ピノキオピー 63.99% / すりぃ 11.85% / cosMo@暴走P 4.56% |
| 115 | ピノキオピー | すきなことだけでいいです | cosMo@暴走P | 2 | Wrong | Rejected | 42.84% | cosMo@暴走P 42.84% / ピノキオピー 13.32% / sasakure.UK 8.20% |
| 116 | ピノキオピー | ぼくらはみんな意味不明 | DECO*27 | 2 | Wrong | Rejected | 25.70% | DECO*27 25.70% / ピノキオピー 20.96% / すりぃ 14.52% |
| 117 | 煮ル果実 | アイアルの勘違い | 煮ル果実 | 1 | Correct | Accepted | 92.69% | 煮ル果実 92.69% / Chinozo 2.40% / すりぃ 1.56% |
| 118 | 煮ル果実 | ドクトリーヌ | 煮ル果実 | 1 | Correct | Accepted | 86.29% | 煮ル果実 86.29% / Chinozo 5.07% / R Sound Design 4.31% |
| 119 | 煮ル果実 | ハングリーニコル | 煮ル果実 | 1 | Correct | Accepted | 80.75% | 煮ル果実 80.75% / じん 8.36% / Chinozo 7.70% |
| 120 | 煮ル果実 | 命辛辛 | 煮ル果実 | 1 | Correct | Accepted | 92.61% | 煮ル果実 92.61% / てにをは 3.26% / Chinozo 1.72% |
| 121 | 稲葉曇 | ひみつの小学生 | 稲葉曇 | 1 | Correct | Accepted | 99.98% | 稲葉曇 99.98% / かいりきベア 0.01% / じん 0.00% |
| 122 | 稲葉曇 | アンチサイクロン | 稲葉曇 | 1 | Correct | Accepted | 99.94% | 稲葉曇 99.94% / じん 0.02% / かいりきベア 0.01% |
| 123 | 稲葉曇 | フロートプレイ | 稲葉曇 | 1 | Correct | Accepted | 99.93% | 稲葉曇 99.93% / r-906 0.03% / n-buna 0.01% |
| 124 | 稲葉曇 | 電気予報 | 稲葉曇 | 1 | Correct | Accepted | 85.96% | 稲葉曇 85.96% / じん 4.50% / cosMo@暴走P 1.90% |

## Development holdout: all 62 songs

| # | True producer | Song | Model Top-1 | True rank | Top-1 | Accept state | Confidence | Top-3 |
|---:|---|---|---|---:|---|---|---:|---|
| 1 | 40mP | Color of Drops | 40mP | 1 | Correct | Rejected | 61.53% | 40mP 61.53% / じん 12.59% / とあ 12.57% |
| 2 | 40mP | Snow Fairy Story | 40mP | 1 | Correct | Accepted | 84.44% | 40mP 84.44% / すりぃ 7.62% / Neru 2.29% |
| 3 | Ayase | HERO | Ayase | 1 | Correct | Accepted | 84.84% | Ayase 84.84% / kemu 10.99% / はるまきごはん 1.24% |
| 4 | Ayase | アイドル | DECO*27 | 17 | Wrong | Accepted | 83.10% | DECO*27 83.10% / じん 6.83% / すりぃ 4.61% |
| 5 | Chinozo | タクシィ | Chinozo | 1 | Correct | Accepted | 98.50% | Chinozo 98.50% / てにをは 0.64% / MIMI 0.38% |
| 6 | Chinozo | 武装乙女 | Chinozo | 1 | Correct | Accepted | 90.46% | Chinozo 90.46% / Kanaria 5.11% / すりぃ 2.43% |
| 7 | cosMo@暴走P | ギザバ怪文書 | cosMo@暴走P | 1 | Correct | Accepted | 95.92% | cosMo@暴走P 95.92% / sasakure.UK 1.68% / はるまきごはん 0.67% |
| 8 | cosMo@暴走P | ダイジョブですか？ | cosMo@暴走P | 1 | Correct | Accepted | 95.04% | cosMo@暴走P 95.04% / DECO*27 2.91% / sasakure.UK 0.88% |
| 9 | DECO*27 | モザイクロール | DECO*27 | 1 | Correct | Rejected | 20.69% | DECO*27 20.69% / Orangestar 16.07% / Ayase 11.43% |
| 10 | DECO*27 | 二息歩行 | sasakure.UK | 2 | Wrong | Rejected | 22.63% | sasakure.UK 22.63% / DECO*27 17.58% / じん 11.08% |
| 11 | Kanaria | チャンピオン | cosMo@暴走P | 7 | Wrong | Rejected | 21.02% | cosMo@暴走P 21.02% / DECO*27 15.89% / すりぃ 13.26% |
| 12 | Kanaria | 百鬼祭 | Kanaria | 1 | Correct | Rejected | 59.22% | Kanaria 59.22% / はるまきごはん 14.97% / ハチ 8.17% |
| 13 | kemu | 88☆彡 | DECO*27 | 8 | Wrong | Rejected | 47.93% | DECO*27 47.93% / すりぃ 23.73% / はるまきごはん 7.83% |
| 14 | kemu | セカイ | 40mP | 20 | Wrong | Rejected | 34.64% | 40mP 34.64% / DECO*27 32.18% / じん 17.56% |
| 15 | MARETU | スヂ | MARETU | 1 | Correct | Accepted | 97.88% | MARETU 97.88% / DECO*27 0.62% / すりぃ 0.55% |
| 16 | MARETU | 少女ケシゴム | MARETU | 1 | Correct | Accepted | 88.24% | MARETU 88.24% / かいりきベア 5.80% / じん 1.91% |
| 17 | MIMI | Pale | MIMI | 1 | Correct | Accepted | 98.42% | MIMI 98.42% / Ayase 0.41% / すりぃ 0.41% |
| 18 | MIMI | カラバコにアイ | MIMI | 1 | Correct | Rejected | 40.92% | MIMI 40.92% / Ayase 24.10% / すりぃ 18.93% |
| 19 | Mitchie M | イージーデンス | Mitchie M | 1 | Correct | Accepted | 96.20% | Mitchie M 96.20% / 煮ル果実 1.07% / R Sound Design 1.02% |
| 20 | Mitchie M | ワーワーワールド | Mitchie M | 1 | Correct | Accepted | 96.40% | Mitchie M 96.40% / 煮ル果実 1.34% / DECO*27 0.94% |
| 21 | n-buna | 劇場愛歌 | n-buna | 1 | Correct | Accepted | 82.95% | n-buna 82.95% / Neru 11.16% / じん 4.80% |
| 22 | n-buna | 言って。 | じん | 2 | Wrong | Accepted | 63.13% | じん 63.13% / n-buna 20.22% / Neru 9.06% |
| 23 | Neru | 命のユースティティア | Neru | 1 | Correct | Rejected | 56.55% | Neru 56.55% / すりぃ 19.23% / kemu 7.72% |
| 24 | Neru | 少年少女カメレオンシンプトム | Neru | 1 | Correct | Accepted | 98.92% | Neru 98.92% / じん 0.56% / kemu 0.22% |
| 25 | Orangestar | イヤホンと蝉時雨 | Orangestar | 1 | Correct | Accepted | 99.70% | Orangestar 99.70% / cosMo@暴走P 0.05% / ハチ 0.04% |
| 26 | Orangestar | スターナイトスノウ | n-buna | 5 | Wrong | Rejected | 58.67% | n-buna 58.67% / Neru 25.19% / すりぃ 6.61% |
| 27 | R Sound Design | EDIBLE | R Sound Design | 1 | Correct | Accepted | 69.27% | R Sound Design 69.27% / じん 15.16% / すりぃ 6.58% |
| 28 | R Sound Design | レテノール | R Sound Design | 1 | Correct | Accepted | 99.67% | R Sound Design 99.67% / cosMo@暴走P 0.10% / はるまきごはん 0.10% |
| 29 | r-906 | Catchy !? | r-906 | 1 | Correct | Rejected | 47.63% | r-906 47.63% / てにをは 15.17% / じん 10.84% |
| 30 | r-906 | JUMPIN’ OVER ! | r-906 | 1 | Correct | Accepted | 88.83% | r-906 88.83% / DECO*27 2.59% / じん 1.67% |
| 31 | sasakure.UK | アフターエポックス | sasakure.UK | 1 | Correct | Accepted | 84.11% | sasakure.UK 84.11% / はるまきごはん 4.35% / Kanaria 4.02% |
| 32 | sasakure.UK | ツギハギエデン | sasakure.UK | 1 | Correct | Accepted | 96.69% | sasakure.UK 96.69% / R Sound Design 1.33% / 煮ル果実 0.73% |
| 33 | syudou | うっせぇわ | すりぃ | 5 | Wrong | Accepted | 87.03% | すりぃ 87.03% / てにをは 2.15% / Chinozo 1.78% |
| 34 | syudou | アメイジングハッピーハロウィンナイト | syudou | 1 | Correct | Rejected | 55.78% | syudou 55.78% / すりぃ 18.74% / MARETU 5.88% |
| 35 | wowaka | and I'm home | とあ | 29 | Wrong | Rejected | 33.04% | とあ 33.04% / じん 20.25% / すりぃ 11.50% |
| 36 | wowaka | 少女について | wowaka | 1 | Correct | Rejected | 40.14% | wowaka 40.14% / てにをは 21.05% / 煮ル果実 15.02% |
| 37 | いよわ | たぶん終わり | いよわ | 1 | Correct | Accepted | 97.02% | いよわ 97.02% / Ayase 1.27% / r-906 0.62% |
| 38 | いよわ | オーバー！ | いよわ | 1 | Correct | Accepted | 93.00% | いよわ 93.00% / cosMo@暴走P 3.80% / じん 0.79% |
| 39 | かいりきベア | アイ情劣等生 | かいりきベア | 1 | Correct | Accepted | 90.04% | かいりきベア 90.04% / じん 4.59% / Neru 2.15% |
| 40 | かいりきベア | アナタサマ | かいりきベア | 1 | Correct | Accepted | 97.35% | かいりきベア 97.35% / すりぃ 0.88% / Chinozo 0.78% |
| 41 | じん | コノハの世界事情 | じん | 1 | Correct | Accepted | 96.14% | じん 96.14% / cosMo@暴走P 2.37% / Neru 0.61% |
| 42 | じん | 夕景イエスタデイ | じん | 1 | Correct | Accepted | 78.66% | じん 78.66% / Neru 7.65% / てにをは 3.66% |
| 43 | すりぃ | ノルア・ドルア・エー | すりぃ | 1 | Correct | Accepted | 69.47% | すりぃ 69.47% / じん 16.86% / DECO*27 2.07% |
| 44 | すりぃ | メンヘラ取扱説明書 | kemu | 7 | Wrong | Accepted | 67.89% | kemu 67.89% / DECO*27 11.42% / じん 8.42% |
| 45 | てにをは | ギラギラ | てにをは | 1 | Correct | Rejected | 37.39% | てにをは 37.39% / すりぃ 22.35% / R Sound Design 11.09% |
| 46 | てにをは | 名探偵連続殺人事件 | 煮ル果実 | 7 | Wrong | Rejected | 22.28% | 煮ル果実 22.28% / DECO*27 15.34% / すりぃ 12.00% |
| 47 | とあ | さよならスーヴェニア | とあ | 1 | Correct | Accepted | 96.83% | とあ 96.83% / すりぃ 1.41% / じん 0.43% |
| 48 | とあ | アイシテ | とあ | 1 | Correct | Rejected | 50.14% | とあ 50.14% / すりぃ 8.23% / ハチ 8.07% |
| 49 | なきそ | 今は限り | なきそ | 1 | Correct | Accepted | 99.91% | なきそ 99.91% / r-906 0.08% / kemu 0.00% |
| 50 | なきそ | 惡ふざけ | cosMo@暴走P | 2 | Wrong | Rejected | 32.42% | cosMo@暴走P 32.42% / なきそ 14.55% / じん 14.01% |
| 51 | はるまきごはん | ディナーベル | はるまきごはん | 1 | Correct | Accepted | 89.15% | はるまきごはん 89.15% / MIMI 4.12% / Ayase 1.35% |
| 52 | はるまきごはん | 蛍はいなかった | はるまきごはん | 1 | Correct | Accepted | 89.91% | はるまきごはん 89.91% / すりぃ 2.04% / じん 1.98% |
| 53 | ナユタン星人 | オペラ！スペースオペラ！  | cosMo@暴走P | 28 | Wrong | Accepted | 67.63% | cosMo@暴走P 67.63% / じん 12.29% / DECO*27 11.22% |
| 54 | ナユタン星人 | モア！ジャンプ！モア！ | すりぃ | 12 | Wrong | Rejected | 32.72% | すりぃ 32.72% / Neru 22.71% / n-buna 19.20% |
| 55 | ハチ | clock lock works | ハチ | 1 | Correct | Accepted | 99.86% | ハチ 99.86% / とあ 0.03% / sasakure.UK 0.03% |
| 56 | ハチ | KICK BACK | じん | 12 | Wrong | Accepted | 67.78% | じん 67.78% / すりぃ 12.70% / DECO*27 12.38% |
| 57 | ピノキオピー | アップルドットコム | ピノキオピー | 1 | Correct | Accepted | 81.02% | ピノキオピー 81.02% / すりぃ 9.13% / Chinozo 3.67% |
| 58 | ピノキオピー | 魔法少女とチョコレゐト | ピノキオピー | 1 | Correct | Accepted | 99.79% | ピノキオピー 99.79% / Chinozo 0.07% / DECO*27 0.05% |
| 59 | 煮ル果実 | イヱスマン | 煮ル果実 | 1 | Correct | Accepted | 83.89% | 煮ル果実 83.89% / じん 8.77% / いよわ 2.14% |
| 60 | 煮ル果実 | ヘイヴン | 煮ル果実 | 1 | Correct | Accepted | 92.10% | 煮ル果実 92.10% / てにをは 3.66% / DECO*27 1.48% |
| 61 | 稲葉曇 | パスカルビーツ | 稲葉曇 | 1 | Correct | Accepted | 96.73% | 稲葉曇 96.73% / かいりきベア 2.49% / てにをは 0.18% |
| 62 | 稲葉曇 | ループスピナ | 稲葉曇 | 1 | Correct | Accepted | 64.63% | 稲葉曇 64.63% / かいりきベア 24.49% / n-buna 3.54% |
