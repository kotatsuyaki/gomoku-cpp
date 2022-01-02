#!/usr/bin/env bash

while true; do
    GEN=$(<gen)
    NEWGEN=$(( $GEN + 1 ))

    date | tee -a train.log
    echo "$(tput bold)(train.sh) Training gen $NEWGEN$(tput sgr0)" | tee -a train.log
    ./main train | tee -a train.log

    date | tee -a train.log
    echo "$(tput bold)(train.sh) Done training gen $NEWGEN$(tput sgr0)" | tee -a train.log

    echo "$NEWGEN" > gen

    cp "net.pt" "net.$NEWGEN.pt"
    cp "opt.pt" "opt.$NEWGEN.pt"
done
