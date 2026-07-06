# jepa-vlm run comparison (latest logged step)

| run | step | loss | reg_loss | mtp_loss | target_std | adj_cos | copy_mse | sec_per_step | val_reg_loss | val_target_std | val_adj_cos |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen_vit | 6000 | 0.29916 | 0.16551 | 0.13366 | 0.41866 | 0.9587 | 0.1058 | 0.60194 | None | None | None |
| mask75 | 6000 | 0.07354 | 0.04698 | 0.02656 | 0.22777 | 0.99298 | 0.01972 | 1.1549 | None | None | None |
| mtp_off | 6000 | 0.03916 | 0.03916 | None | 0.22036 | 0.98838 | 0.03074 | 1.13421 | None | None | None |
| v1 | 6000 | 0.03294 | 0.01127 | 0.02167 | 0.28012 | 0.99316 | None | 1.12857 | None | None | None |
| v21 | 6000 | 0.06373 | 0.03586 | 0.02787 | 0.21014 | 0.99316 | 0.01735 | 1.14449 | None | None | None |
