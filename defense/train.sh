EPOCHS=20
GPUS='3'
OPTIM='adam'
GRATIO=3
BN='none'

SAVE='res'_'g'${GRATIO}_${OPTIM}_'238k'_'UDenoiser_FLOSS'

python main.py  \
    --save  ${SAVE}   \
    --gpus ${GPUS}   \
    --gratio ${GRATIO}   \
    --optim ${OPTIM}      \
    --bn_type ${BN}      \
