#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
BEGIN {
    printf("beam_width, beam_length, average_num_accepted_tokens, tokens_per_sec\n");
}
/Title: beamwidth/ {
    nc = gensub(/^.*beamwidth([0-9]+)_beamlen[0-9]+.*$/, "\\1", 1);
    cl = gensub(/^.*beamwidth[0-9]+_beamlen([0-9]+).*$/, "\\1", 1);
    printf("%d, %d, ", nc, cl);
}

/Average number of tokens per step:/ {
    anat = gensub(/^Average number of tokens per step: +(.*)$/, "\\1", 1);
    printf("%f, ", anat);
}
/Tokens\/second/ {
    tps = gensub(/^Tokens\/second +(.*)$/, "\\1", 1);
    printf("%s\n", tps);
}
