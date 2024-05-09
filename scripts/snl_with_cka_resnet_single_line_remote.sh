python3 snl_finetune_unstructured_with_cka.py "$DATASET" "$ARCH" "$OUTDIR" "$SAVEDIR" --relu_budget $RELU_BUDGET --alpha $ALPHA  --alpha_cka $ALPHA_CKA --lr $LR --threshold $THRESHOLD --batch $BATCH --logname "$LOGNAME" --finetune_epochs $FINETUNE_EPOCH

