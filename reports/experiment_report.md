# Relatório de experimentos — MovieLens 100K

## Dados e configurações

Avaliação feita no MovieLens 100K local, usando os cinco pares oficiais `u1.base/u1.test` até `u5.base/u5.test`. Em cada fold, o modelo é construído somente com o arquivo `.base`; o `.test` fornece relevância. O catálogo completo de `u.item` permanece disponível, mesmo para filmes sem avaliações no treino.

- Filmes no catálogo: **1682**.
- Gêneros no catálogo: **19**.
- Usuários: **943**.
- K: **10**; relevância: **test rating >= 4**.
- Variantes top-K: **[20, 50, 100] vizinhos**.
- Bootstrap pareado: **10,000** reamostragens; semente **42**; IC percentil de 95%.

| Fold | Avaliações de treino | Avaliações de teste | Usuários no teste |
|---:|---:|---:|---:|
| 1 | 80000 | 20000 | 459 |
| 2 | 80000 | 20000 | 653 |
| 3 | 80000 | 20000 | 869 |
| 4 | 80000 | 20000 | 923 |
| 5 | 80000 | 20000 | 927 |

## Metodologia e métricas

Precision@10 usa todos os usuários do teste e divide acertos por 10; listas menores contam posições ausentes como não acertos. Recall@10 e NDCG@10 omitem usuários sem itens relevantes. Itens relevantes são ratings de teste >= 4; ausência de rating não é tratada como nota negativa. Cobertura é a fração do catálogo que aparece nas listas. Diversidade é a média macro por usuário de `1 - Jaccard` dos gêneros; pares sem gêneros disponíveis não são comparáveis. Novidade é a média de `-log2(freqência no treino / ratings de treino)`. As médias por fold são macro, com peso igual entre folds definidos.

## Item-Based atual e baseline de popularidade

A popularidade é contada exclusivamente em cada `.base`; recomendações excluem filmes já avaliados pelo usuário nesse treino.

| Fold | Modelo | Usuários (sem relevantes) | P@10 | R@10 | NDCG@10 | Cobertura | Diversidade | Novidade |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Item-Based | 459 (3) | 0.2538 | 0.1203 | 0.3023 | 0.2622 | 0.7847 | 9.5306 |
| 2 | Item-Based | 653 (9) | 0.2035 | 0.1355 | 0.2529 | 0.2788 | 0.7806 | 9.5874 |
| 3 | Item-Based | 869 (20) | 0.1581 | 0.1471 | 0.2125 | 0.2985 | 0.7747 | 9.7877 |
| 4 | Item-Based | 923 (33) | 0.1465 | 0.1439 | 0.2026 | 0.2979 | 0.7799 | 9.8871 |
| 5 | Item-Based | 927 (49) | 0.1451 | 0.1538 | 0.2140 | 0.3026 | 0.7819 | 9.8839 |
| Macro | Item-Based | média macro por fold | 0.1814 | 0.1401 | 0.2369 | 0.2880 | 0.7804 | 9.7353 |
| 1 | Popularidade | 459 (3) | 0.2122 | 0.1180 | 0.2451 | 0.0303 | 0.8411 | 7.7748 |
| 2 | Popularidade | 653 (9) | 0.1737 | 0.1449 | 0.2192 | 0.0309 | 0.8340 | 7.8136 |
| 3 | Popularidade | 869 (20) | 0.1315 | 0.1353 | 0.1807 | 0.0351 | 0.8354 | 7.8351 |
| 4 | Popularidade | 923 (33) | 0.1281 | 0.1453 | 0.1771 | 0.0404 | 0.8349 | 7.8511 |
| 5 | Popularidade | 927 (49) | 0.1204 | 0.1371 | 0.1688 | 0.0392 | 0.8362 | 7.8525 |
| Macro | Popularidade | média macro por fold | 0.1532 | 0.1361 | 0.1982 | 0.0352 | 0.8363 | 7.8254 |

## Variantes Item-Based

A variante sem centralização usa cosseno nos ratings originais. As variantes top-K limitam vizinhos positivos por filme candidato. O Item-Based centralizado sem limite permanece como referência.

| Variante | Fold | Precision@10 | Recall@10 | NDCG@10 | Cobertura | Diversidade | Novidade |
|---|---:|---:|---:|---:|---:|---:|---:|
| centered_all_neighbors | 1 | 0.2538 | 0.1203 | 0.3023 | 0.2622 | 0.7847 | 9.5306 |
| centered_all_neighbors | 2 | 0.2035 | 0.1355 | 0.2529 | 0.2788 | 0.7806 | 9.5874 |
| centered_all_neighbors | 3 | 0.1581 | 0.1471 | 0.2125 | 0.2985 | 0.7747 | 9.7877 |
| centered_all_neighbors | 4 | 0.1465 | 0.1439 | 0.2026 | 0.2979 | 0.7799 | 9.8871 |
| centered_all_neighbors | 5 | 0.1451 | 0.1538 | 0.2140 | 0.3026 | 0.7819 | 9.8839 |
| **centered_all_neighbors — macro** | — | 0.1814 | 0.1401 | 0.2369 | 0.2880 | 0.7804 | 9.7353 |
| uncentered_all_neighbors | 1 | 0.3290 | 0.2059 | 0.3990 | 0.1058 | 0.7956 | 8.2677 |
| uncentered_all_neighbors | 2 | 0.2668 | 0.2271 | 0.3562 | 0.1112 | 0.7939 | 8.2857 |
| uncentered_all_neighbors | 3 | 0.2212 | 0.2444 | 0.3103 | 0.1177 | 0.8107 | 8.3156 |
| uncentered_all_neighbors | 4 | 0.2086 | 0.2307 | 0.3063 | 0.1272 | 0.8175 | 8.3289 |
| uncentered_all_neighbors | 5 | 0.1983 | 0.2410 | 0.3054 | 0.1266 | 0.8223 | 8.3208 |
| **uncentered_all_neighbors — macro** | — | 0.2448 | 0.2298 | 0.3355 | 0.1177 | 0.8080 | 8.3037 |
| centered_top_20_neighbors | 1 | 0.2444 | 0.1183 | 0.2945 | 0.2646 | 0.7875 | 9.6497 |
| centered_top_20_neighbors | 2 | 0.1802 | 0.1273 | 0.2263 | 0.2729 | 0.7897 | 10.0314 |
| centered_top_20_neighbors | 3 | 0.1348 | 0.1341 | 0.1876 | 0.2961 | 0.7874 | 10.2070 |
| centered_top_20_neighbors | 4 | 0.1323 | 0.1325 | 0.1880 | 0.2901 | 0.7890 | 10.1667 |
| centered_top_20_neighbors | 5 | 0.1284 | 0.1388 | 0.1916 | 0.3020 | 0.7854 | 10.2689 |
| **centered_top_20_neighbors — macro** | — | 0.1640 | 0.1302 | 0.2176 | 0.2851 | 0.7878 | 10.0647 |
| centered_top_50_neighbors | 1 | 0.2495 | 0.1194 | 0.2999 | 0.2622 | 0.7863 | 9.5565 |
| centered_top_50_neighbors | 2 | 0.1930 | 0.1324 | 0.2425 | 0.2735 | 0.7820 | 9.7183 |
| centered_top_50_neighbors | 3 | 0.1467 | 0.1429 | 0.1981 | 0.2919 | 0.7786 | 9.9684 |
| centered_top_50_neighbors | 4 | 0.1387 | 0.1389 | 0.1928 | 0.2973 | 0.7833 | 10.0311 |
| centered_top_50_neighbors | 5 | 0.1367 | 0.1468 | 0.2028 | 0.2979 | 0.7838 | 10.0457 |
| **centered_top_50_neighbors — macro** | — | 0.1729 | 0.1361 | 0.2272 | 0.2845 | 0.7828 | 9.8640 |
| centered_top_100_neighbors | 1 | 0.2538 | 0.1203 | 0.3023 | 0.2604 | 0.7852 | 9.5337 |
| centered_top_100_neighbors | 2 | 0.2028 | 0.1353 | 0.2520 | 0.2788 | 0.7810 | 9.6030 |
| centered_top_100_neighbors | 3 | 0.1554 | 0.1458 | 0.2099 | 0.2979 | 0.7757 | 9.8242 |
| centered_top_100_neighbors | 4 | 0.1443 | 0.1426 | 0.1998 | 0.2973 | 0.7812 | 9.9355 |
| centered_top_100_neighbors | 5 | 0.1422 | 0.1518 | 0.2103 | 0.3032 | 0.7828 | 9.9396 |
| **centered_top_100_neighbors — macro** | — | 0.1797 | 0.1392 | 0.2348 | 0.2875 | 0.7812 | 9.7672 |

## Resultados segmentados por histórico de treino

Cada usuário é classificado pela contagem de avaliações em `split.train`: 1–5, 6–10, 11–20 ou 21+. Precision inclui todos os usuários; Recall e NDCG omitem os sem positivos. A cobertura de cada segmento usa o catálogo completo.

| Modelo | Histórico | Folds com usuários/relevantes | Usuários médios por fold | P@10 | R@10 | NDCG@10 | Cobertura |
|---|---|---:|---:|---:|---:|---:|---:|
| item_based_centered | 1-5 | 2/5 / 2/5 | 0.40 | 0.1500 | 0.2917 | 0.2529 | 0.0059 |
| item_based_centered | 6-10 | 5/5 / 5/5 | 9.20 | 0.0563 | 0.0968 | 0.1193 | 0.0392 |
| item_based_centered | 11-20 | 5/5 / 5/5 | 122.00 | 0.0625 | 0.1441 | 0.1331 | 0.1813 |
| item_based_centered | 21+ | 5/5 / 5/5 | 634.60 | 0.2099 | 0.1395 | 0.2616 | 0.2278 |
| item_based_uncentered | 1-5 | 2/5 / 2/5 | 0.40 | 0.2000 | 0.3333 | 0.3626 | 0.0059 |
| item_based_uncentered | 6-10 | 5/5 / 5/5 | 9.20 | 0.1834 | 0.3956 | 0.3087 | 0.0264 |
| item_based_uncentered | 11-20 | 5/5 / 5/5 | 122.00 | 0.1382 | 0.3177 | 0.2810 | 0.0700 |
| item_based_uncentered | 21+ | 5/5 / 5/5 | 634.60 | 0.2693 | 0.2099 | 0.3485 | 0.1109 |
| popularity | 1-5 | 2/5 / 2/5 | 0.40 | 0.0500 | 0.0417 | 0.1100 | 0.0059 |
| popularity | 6-10 | 5/5 / 5/5 | 9.20 | 0.0748 | 0.1055 | 0.0825 | 0.0077 |
| popularity | 11-20 | 5/5 / 5/5 | 122.00 | 0.0752 | 0.1854 | 0.1385 | 0.0102 |
| popularity | 21+ | 5/5 / 5/5 | 634.60 | 0.1719 | 0.1266 | 0.2134 | 0.0352 |

### Resultados segmentados por fold

| Fold | Modelo | Histórico | Usuários (sem relevantes) | P@10 | R@10 | NDCG@10 | Itens distintos / catálogo | Cobertura |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | item_based_centered | 1-5 | 1 (0) | 0.1000 | 0.0833 | 0.2201 | 10 / 1682 | 0.0059 |
| 1 | item_based_centered | 6-10 | 20 (0) | 0.0850 | 0.1078 | 0.1279 | 132 / 1682 | 0.0785 |
| 1 | item_based_centered | 11-20 | 96 (0) | 0.0760 | 0.1301 | 0.1465 | 288 / 1682 | 0.1712 |
| 1 | item_based_centered | 21+ | 342 (3) | 0.3140 | 0.1183 | 0.3570 | 302 / 1682 | 0.1795 |
| 1 | item_based_uncentered | 1-5 | 1 (0) | 0.2000 | 0.1667 | 0.2837 | 10 / 1682 | 0.0059 |
| 1 | item_based_uncentered | 6-10 | 20 (0) | 0.2100 | 0.3147 | 0.3058 | 70 / 1682 | 0.0416 |
| 1 | item_based_uncentered | 11-20 | 96 (0) | 0.1802 | 0.3174 | 0.3020 | 126 / 1682 | 0.0749 |
| 1 | item_based_uncentered | 21+ | 342 (3) | 0.3781 | 0.1680 | 0.4324 | 154 / 1682 | 0.0916 |
| 1 | popularity | 1-5 | 1 (0) | 0.1000 | 0.0833 | 0.2201 | 10 / 1682 | 0.0059 |
| 1 | popularity | 6-10 | 20 (0) | 0.1050 | 0.1376 | 0.1283 | 14 / 1682 | 0.0083 |
| 1 | popularity | 11-20 | 96 (0) | 0.0896 | 0.1538 | 0.1367 | 17 / 1682 | 0.0101 |
| 1 | popularity | 21+ | 342 (3) | 0.2532 | 0.1068 | 0.2827 | 51 / 1682 | 0.0303 |
| 2 | item_based_centered | 1-5 | 0 (0) | — | — | — | 0 / 1682 | — |
| 2 | item_based_centered | 6-10 | 4 (0) | 0.0750 | 0.1708 | 0.2136 | 37 / 1682 | 0.0220 |
| 2 | item_based_centered | 11-20 | 113 (4) | 0.0584 | 0.1406 | 0.1128 | 282 / 1682 | 0.1677 |
| 2 | item_based_centered | 21+ | 536 (5) | 0.2351 | 0.1343 | 0.2819 | 374 / 1682 | 0.2224 |
| 2 | item_based_uncentered | 1-5 | 0 (0) | — | — | — | 0 / 1682 | — |
| 2 | item_based_uncentered | 6-10 | 4 (0) | 0.2500 | 0.5000 | 0.4144 | 33 / 1682 | 0.0196 |
| 2 | item_based_uncentered | 11-20 | 113 (4) | 0.1274 | 0.3232 | 0.2839 | 113 / 1682 | 0.0672 |
| 2 | item_based_uncentered | 21+ | 536 (5) | 0.2963 | 0.2053 | 0.3706 | 179 / 1682 | 0.1064 |
| 2 | popularity | 1-5 | 0 (0) | — | — | — | 0 / 1682 | — |
| 2 | popularity | 6-10 | 4 (0) | 0.1000 | 0.1936 | 0.1143 | 15 / 1682 | 0.0089 |
| 2 | popularity | 11-20 | 113 (4) | 0.0717 | 0.1910 | 0.1296 | 16 / 1682 | 0.0095 |
| 2 | popularity | 21+ | 536 (5) | 0.1957 | 0.1351 | 0.2384 | 52 / 1682 | 0.0309 |
| 3 | item_based_centered | 1-5 | 0 (0) | — | — | — | 0 / 1682 | — |
| 3 | item_based_centered | 6-10 | 2 (1) | 0.0000 | 0.0000 | 0.0000 | 20 / 1682 | 0.0119 |
| 3 | item_based_centered | 11-20 | 137 (10) | 0.0562 | 0.1609 | 0.1410 | 320 / 1682 | 0.1902 |
| 3 | item_based_centered | 21+ | 730 (9) | 0.1777 | 0.1449 | 0.2254 | 409 / 1682 | 0.2432 |
| 3 | item_based_uncentered | 1-5 | 0 (0) | — | — | — | 0 / 1682 | — |
| 3 | item_based_uncentered | 6-10 | 2 (1) | 0.1000 | 0.6667 | 0.2837 | 18 / 1682 | 0.0107 |
| 3 | item_based_uncentered | 11-20 | 137 (10) | 0.1117 | 0.3222 | 0.2709 | 112 / 1682 | 0.0666 |
| 3 | item_based_uncentered | 21+ | 730 (9) | 0.2421 | 0.2301 | 0.3173 | 192 / 1682 | 0.1141 |
| 3 | popularity | 1-5 | 0 (0) | — | — | — | 0 / 1682 | — |
| 3 | popularity | 6-10 | 2 (1) | 0.0000 | 0.0000 | 0.0000 | 11 / 1682 | 0.0065 |
| 3 | popularity | 11-20 | 137 (10) | 0.0664 | 0.1949 | 0.1405 | 17 / 1682 | 0.0101 |
| 3 | popularity | 21+ | 730 (9) | 0.1441 | 0.1250 | 0.1881 | 59 / 1682 | 0.0351 |
| 4 | item_based_centered | 1-5 | 0 (0) | — | — | — | 0 / 1682 | — |
| 4 | item_based_centered | 6-10 | 6 (0) | 0.0500 | 0.1074 | 0.1487 | 59 / 1682 | 0.0351 |
| 4 | item_based_centered | 11-20 | 140 (13) | 0.0486 | 0.1333 | 0.1130 | 321 / 1682 | 0.1908 |
| 4 | item_based_centered | 21+ | 777 (20) | 0.1649 | 0.1459 | 0.2181 | 405 / 1682 | 0.2408 |
| 4 | item_based_uncentered | 1-5 | 0 (0) | — | — | — | 0 / 1682 | — |
| 4 | item_based_uncentered | 6-10 | 6 (0) | 0.1500 | 0.2296 | 0.2295 | 43 / 1682 | 0.0256 |
| 4 | item_based_uncentered | 11-20 | 140 (13) | 0.1314 | 0.3264 | 0.2773 | 121 / 1682 | 0.0719 |
| 4 | item_based_uncentered | 21+ | 777 (20) | 0.2229 | 0.2146 | 0.3118 | 203 / 1682 | 0.1207 |
| 4 | popularity | 1-5 | 0 (0) | — | — | — | 0 / 1682 | — |
| 4 | popularity | 6-10 | 6 (0) | 0.0833 | 0.0907 | 0.0791 | 12 / 1682 | 0.0071 |
| 4 | popularity | 11-20 | 140 (13) | 0.0757 | 0.2245 | 0.1554 | 18 / 1682 | 0.0107 |
| 4 | popularity | 21+ | 777 (20) | 0.1378 | 0.1324 | 0.1815 | 68 / 1682 | 0.0404 |
| 5 | item_based_centered | 1-5 | 1 (0) | 0.2000 | 0.5000 | 0.2856 | 10 / 1682 | 0.0059 |
| 5 | item_based_centered | 6-10 | 14 (0) | 0.0714 | 0.0979 | 0.1065 | 82 / 1682 | 0.0488 |
| 5 | item_based_centered | 11-20 | 124 (9) | 0.0734 | 0.1558 | 0.1523 | 314 / 1682 | 0.1867 |
| 5 | item_based_centered | 21+ | 788 (40) | 0.1576 | 0.1541 | 0.2254 | 426 / 1682 | 0.2533 |
| 5 | item_based_uncentered | 1-5 | 1 (0) | 0.2000 | 0.5000 | 0.4415 | 10 / 1682 | 0.0059 |
| 5 | item_based_uncentered | 6-10 | 14 (0) | 0.2071 | 0.2671 | 0.3103 | 58 / 1682 | 0.0345 |
| 5 | item_based_uncentered | 11-20 | 124 (9) | 0.1403 | 0.2992 | 0.2708 | 117 / 1682 | 0.0696 |
| 5 | item_based_uncentered | 21+ | 788 (40) | 0.2072 | 0.2312 | 0.3104 | 205 / 1682 | 0.1219 |
| 5 | popularity | 1-5 | 1 (0) | 0.0000 | 0.0000 | 0.0000 | 10 / 1682 | 0.0059 |
| 5 | popularity | 6-10 | 14 (0) | 0.0857 | 0.1054 | 0.0908 | 13 / 1682 | 0.0077 |
| 5 | popularity | 11-20 | 124 (9) | 0.0726 | 0.1630 | 0.1304 | 18 / 1682 | 0.0107 |
| 5 | popularity | 21+ | 788 (40) | 0.1287 | 0.1339 | 0.1763 | 66 / 1682 | 0.0392 |

## Comparação pareada e bootstrap

Diferença = variante sem centralização menos Item-Based centralizado. O bootstrap reamostra usuários com reposição separadamente dentro de cada fold, mantendo os dois resultados do usuário pareados. Os limites usam quantis percentis com interpolação linear. A média macro dá peso igual aos folds com usuários elegíveis.

| Fold | Métrica | Usuários | Diferença média | IC 95% |
|---:|---|---:|---:|---:|
| 1 | precision_at_10 | 459 | 0.07516 | [0.05948; 0.09129] |
| 1 | recall_at_10 | 456 | 0.08564 | [0.06880; 0.10379] |
| 1 | ndcg_at_10 | 456 | 0.09669 | [0.07596; 0.11687] |
| 2 | precision_at_10 | 653 | 0.06325 | [0.05008; 0.07642] |
| 2 | recall_at_10 | 644 | 0.09157 | [0.07528; 0.10790] |
| 2 | ndcg_at_10 | 644 | 0.10332 | [0.08554; 0.12131] |
| 3 | precision_at_10 | 869 | 0.06306 | [0.05305; 0.07330] |
| 3 | recall_at_10 | 849 | 0.09730 | [0.08111; 0.11369] |
| 3 | ndcg_at_10 | 849 | 0.09777 | [0.08325; 0.11209] |
| 4 | precision_at_10 | 923 | 0.06208 | [0.05276; 0.07172] |
| 4 | recall_at_10 | 890 | 0.08680 | [0.07032; 0.10371] |
| 4 | ndcg_at_10 | 890 | 0.10371 | [0.08836; 0.11961] |
| 5 | precision_at_10 | 927 | 0.05318 | [0.04434; 0.06214] |
| 5 | recall_at_10 | 878 | 0.08718 | [0.07174; 0.10283] |
| 5 | ndcg_at_10 | 878 | 0.09139 | [0.07630; 0.10709] |
| **Macro (5/5 folds)** | precision_at_10 | F1: 459, F2: 653, F3: 869, F4: 923, F5: 927 | **0.06335** | — |
| **Macro (5/5 folds)** | recall_at_10 | F1: 456, F2: 644, F3: 849, F4: 890, F5: 878 | **0.08970** | — |
| **Macro (5/5 folds)** | ndcg_at_10 | F1: 456, F2: 644, F3: 849, F4: 890, F5: 878 | **0.09858** | — |

## Limitações e interpretação

Os resultados descrevem somente esta execução e este protocolo no MovieLens 100K. Os cinco folds compartilham usuários e não são amostras independentes; intervalos por fold não provam superioridade universal nem garantem desempenho em outros catálogos. A métrica de relevância binária ignora a diferença entre ratings 4 e 5, e avaliações não observadas não significam desinteresse. Gêneros são metadados amplos, e a novidade depende da distribuição de treino. Resultados indefinidos são exibidos como `—` e como `null` no JSON, nunca como zero.
