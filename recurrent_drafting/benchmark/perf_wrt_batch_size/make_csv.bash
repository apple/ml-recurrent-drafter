#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
set -e # Any command fails, this script fails.
set -o pipefail # Any stage of a pipe fails, the command fails.

echo "batch_size, tps"
for i in {0..6}; do
    batch_size=$((2**i))
    log_file=/tmp/batch_size-"$batch_size".log
    if [[ -f $log_file ]]; then
       if grep "Tokens/second" $log_file > /dev/null; then
	   tps=$(grep "Tokens/second" $log_file | awk '{print $2}')
	   echo "$batch_size, $tps"
       fi
    fi
done
