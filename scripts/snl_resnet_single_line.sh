python3 snl_finetune_unstructured.py "$DATASET" "$ARCH" "$OUTDIR" "$SAVEDIR" --relu_budget $RELU_BUDGET --alpha $ALPHA --lr $LR --threshold $THRESHOLD --batch $BATCH --logname "$LOGNAME" --finetune_epochs $FINETUNE_EPOCH

